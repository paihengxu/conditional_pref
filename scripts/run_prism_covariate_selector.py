#!/usr/bin/env python3
"""Run PRISM covariate-aware feature selection on a completed WIMHF run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parents[1]
WIMHF_REPO = PROJECT_DIR / "repos" / "wimhf"
if str(WIMHF_REPO) not in sys.path:
    sys.path.insert(0, str(WIMHF_REPO))

from wimhf.feature_selection import select_neurons_demeaned_reweighted_lasso
from wimhf.quickstart import (
    get_embeddings,
    load_and_preprocess_dataframe,
    load_config,
    train_sae,
)


DEFAULT_EXACT_PRISM_RUN = (
    PROJECT_DIR
    / "outputs/reproduction/wimhf_exact/"
    / "wimhf_exact_full_prism_gpt5mini_s42_tron_20260608_6985771"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run demeaned-reweighted LASSO over PRISM conversation_type strata."
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=DEFAULT_EXACT_PRISM_RUN,
        help="Completed WIMHF run root containing configs/ and prism/results/.",
    )
    parser.add_argument(
        "--covariate-column",
        default="conversation_type",
        help="PRISM dataframe column defining covariate strata.",
    )
    parser.add_argument(
        "--method",
        default="demeaned-reweighted-lasso",
        choices=["demeaned-reweighted-lasso"],
        help="Covariate-aware selection method.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=[5, 10],
        help="Fixed selected-neuron budgets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to outputs/analysis/prism_covariate_selector/<run_name>.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum iterations for sklearn Lasso.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_DIR / path


def find_config(run_root: Path) -> Path:
    configs = sorted((run_root / "configs").glob("*.json"))
    prism_configs = [path for path in configs if "prism" in path.name]
    if not prism_configs:
        raise FileNotFoundError(f"No PRISM config found under {run_root / 'configs'}")
    if len(prism_configs) > 1:
        print(f"[config] multiple PRISM configs found; using {prism_configs[0]}")
    return prism_configs[0]


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def prepare_config(config_path: Path):
    cfg = load_config(str(config_path))
    cfg.dataset.path = str(resolve_path(Path(cfg.dataset.path)))
    if cfg.runtime.cache_dir is not None:
        cfg.runtime.cache_dir = resolve_path(Path(cfg.runtime.cache_dir))
    if cfg.runtime.checkpoint_dir is not None:
        cfg.runtime.checkpoint_dir = str(resolve_path(Path(cfg.runtime.checkpoint_dir)))
    cfg.runtime.retrain_sae = False
    return cfg


def encode_covariates(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    columns: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, int]]]:
    train_cols = []
    val_cols = []
    codebooks: dict[str, dict[str, int]] = {}
    for col in columns:
        if col not in train_df.columns or col not in val_df.columns:
            raise KeyError(f"Missing covariate column: {col}")
        all_values = pd.concat([train_df[col], val_df[col]], axis=0).astype(str)
        categories = sorted(all_values.dropna().unique().tolist())
        mapping = {value: idx for idx, value in enumerate(categories)}
        codebooks[col] = mapping
        train_cols.append(train_df[col].astype(str).map(mapping).to_numpy())
        val_cols.append(val_df[col].astype(str).map(mapping).to_numpy())
    return np.column_stack(train_cols), np.column_stack(val_cols), codebooks


def fit_eval_model(
    activ_train: np.ndarray,
    y_train: np.ndarray,
    activ_val: np.ndarray,
    y_val: np.ndarray,
    controls_train: np.ndarray,
    controls_val: np.ndarray,
    selected: list[int],
    val_context: pd.Series,
) -> dict[str, Any]:
    X_train = np.concatenate([activ_train[:, selected], controls_train], axis=1)
    X_val = np.concatenate([activ_val[:, selected], controls_val], axis=1)

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_val_std = scaler.transform(X_val)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_std, y_train)

    train_prob = model.predict_proba(X_train_std)[:, 1]
    val_prob = model.predict_proba(X_val_std)[:, 1]
    metrics: dict[str, Any] = {
        "train_auc": float(roc_auc_score(y_train, train_prob)),
        "val_auc": float(roc_auc_score(y_val, val_prob)),
        "per_covariate_val_auc": {},
    }

    for value in sorted(val_context.astype(str).unique()):
        mask = val_context.astype(str).to_numpy() == value
        if len(np.unique(y_val[mask])) < 2:
            metrics["per_covariate_val_auc"][value] = None
        else:
            metrics["per_covariate_val_auc"][value] = float(
                roc_auc_score(y_val[mask], val_prob[mask])
            )
    return metrics


def coefficient_map(payload: dict[str, Any], top_k: int) -> dict[int, float]:
    raw = payload.get(str(top_k), payload.get(top_k, {}))
    return {int(k): float(v) for k, v in raw.items()}


def main() -> None:
    args = parse_args()
    run_root = resolve_path(args.run_root)
    if not run_root.exists():
        raise FileNotFoundError(f"Run root not found: {run_root}")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = PROJECT_DIR / "outputs" / "analysis" / "prism_covariate_selector" / run_root.name
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = find_config(run_root)
    cfg = prepare_config(config_path)
    print(f"[config] {config_path}")
    print(f"[output] {output_dir}")

    (
        train_df,
        val_df,
        train_pred_df,
        val_pred_df,
        dedup_train_df,
        dedup_val_df,
    ) = load_and_preprocess_dataframe(cfg.dataset, cfg.runtime)

    response2embedding, delta_train, delta_val, dedup_delta_train, dedup_delta_val = get_embeddings(
        train_df=train_df,
        val_df=val_df,
        dedup_train_df=dedup_train_df,
        dedup_val_df=dedup_val_df,
        dataset_cfg=cfg.dataset,
        embedding_cfg=cfg.embedding,
        cache_dir=cfg.runtime.cache_dir,
    )
    _ = response2embedding
    sae = train_sae(dedup_delta_train, dedup_delta_val, cfg.sae, cfg.runtime)
    activ_train = sae.get_activations(delta_train)
    activ_val = sae.get_activations(delta_val)

    train_pred_mask = train_df[cfg.dataset.label_column].isin([0, 1]).to_numpy()
    val_pred_mask = val_df[cfg.dataset.label_column].isin([0, 1]).to_numpy()
    activ_train_pred = activ_train[train_pred_mask]
    activ_val_pred = activ_val[val_pred_mask]
    y_train = train_pred_df[cfg.dataset.label_column].to_numpy()
    y_val = val_pred_df[cfg.dataset.label_column].to_numpy()

    controls = cfg.selection.controls
    controls_train = (
        train_pred_df[controls].to_numpy() if controls else np.empty((len(train_pred_df), 0))
    )
    controls_val = val_pred_df[controls].to_numpy() if controls else np.empty((len(val_pred_df), 0))

    cov_columns = [args.covariate_column]
    cov_train, cov_val, codebooks = encode_covariates(train_pred_df, val_pred_df, cov_columns)
    _ = cov_val

    feature_path = run_root / "prism" / "results" / "feature_table.jsonl"
    global_lasso_path = run_root / "prism" / "results" / "lasso_coefficients.json"
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature table not found: {feature_path}")
    if not global_lasso_path.exists():
        raise FileNotFoundError(f"Global LASSO coefficients not found: {global_lasso_path}")

    feature_table = pd.read_json(feature_path, orient="records", lines=True)
    kept_idx = feature_table.loc[feature_table["kept"], "neuron_idx"].astype(int).to_numpy()
    if kept_idx.size == 0:
        raise ValueError("No fidelity-kept neurons found in feature table")
    Z_train_kept = activ_train_pred[:, kept_idx]

    global_lasso = load_json(global_lasso_path)
    covariate_coefficients: dict[str, dict[str, Any]] = {}
    predictive_metrics: dict[str, Any] = {
        "run_root": str(run_root),
        "method": args.method,
        "covariate_columns": cov_columns,
        "controls": controls,
        "top_k": args.top_k,
        "global": {},
        "covariate": {},
    }
    summary_rows: list[dict[str, Any]] = []

    for top_k in sorted(set(args.top_k)):
        n_select = min(top_k, len(kept_idx))
        selected_rel, coefs, info = select_neurons_demeaned_reweighted_lasso(
            activations=Z_train_kept,
            target=y_train,
            n_select=n_select,
            covariates=cov_train,
            max_iter=args.max_iter,
            verbose=args.verbose,
            return_info=True,
        )
        selected_global = kept_idx[np.asarray(selected_rel, dtype=int)].tolist()
        cov_coef_map = {int(idx): float(coef) for idx, coef in zip(selected_global, coefs)}
        covariate_coefficients[str(top_k)] = {
            "selected_neurons": selected_global,
            "coefficients": cov_coef_map,
            "selector_info": info,
        }

        predictive_metrics["covariate"][str(top_k)] = fit_eval_model(
            activ_train=activ_train_pred,
            y_train=y_train,
            activ_val=activ_val_pred,
            y_val=y_val,
            controls_train=controls_train,
            controls_val=controls_val,
            selected=selected_global,
            val_context=val_pred_df[args.covariate_column],
        )

        global_coef_map = coefficient_map(global_lasso, top_k)
        global_selected = list(global_coef_map.keys())
        if global_selected:
            predictive_metrics["global"][str(top_k)] = fit_eval_model(
                activ_train=activ_train_pred,
                y_train=y_train,
                activ_val=activ_val_pred,
                y_val=y_val,
                controls_train=controls_train,
                controls_val=controls_val,
                selected=global_selected,
                val_context=val_pred_df[args.covariate_column],
            )
        else:
            predictive_metrics["global"][str(top_k)] = None

        global_set = set(global_selected)
        cov_set = set(selected_global)
        predictive_metrics["covariate"][str(top_k)]["overlap_with_global"] = sorted(
            global_set & cov_set
        )

        feature_meta = feature_table.set_index("neuron_idx").to_dict(orient="index")
        for method, coef_map, other_set in [
            ("global", global_coef_map, cov_set),
            ("covariate", cov_coef_map, global_set),
        ]:
            ordered = sorted(coef_map.items(), key=lambda item: abs(item[1]), reverse=True)
            for rank, (neuron_idx, coef) in enumerate(ordered, start=1):
                meta = feature_meta.get(int(neuron_idx), {})
                summary_rows.append(
                    {
                        "top_k": top_k,
                        "method": method,
                        "rank": rank,
                        "neuron_idx": int(neuron_idx),
                        "coefficient": float(coef),
                        "selected_by_other_method": int(neuron_idx in other_set),
                        "other_method_coefficient": (
                            cov_coef_map.get(int(neuron_idx))
                            if method == "global"
                            else global_coef_map.get(int(neuron_idx))
                        ),
                        "abbreviated_interpretation": meta.get("abbreviated_interpretation"),
                        "interpretation": meta.get("interpretation"),
                        "prevalence": meta.get("prevalence"),
                        "correlation": meta.get("correlation"),
                        "p_value": meta.get("p_value"),
                        "length_controlled_logit_coef": meta.get("length_controlled_logit_coef"),
                    }
                )

    support = {
        "covariate_columns": cov_columns,
        "codebooks": codebooks,
        "train_counts": {
            str(k): int(v)
            for k, v in train_pred_df[args.covariate_column].astype(str).value_counts().sort_index().items()
        },
        "val_counts": {
            str(k): int(v)
            for k, v in val_pred_df[args.covariate_column].astype(str).value_counts().sort_index().items()
        },
    }

    dump_json(output_dir / "covariate_lasso_coefficients.json", covariate_coefficients)
    dump_json(output_dir / "covariate_predictive_metrics.json", predictive_metrics)
    dump_json(output_dir / "covariate_support.json", support)
    pd.DataFrame(summary_rows).to_csv(output_dir / "covariate_selection_summary.csv", index=False)

    print("[done] wrote:")
    print(f"  {output_dir / 'covariate_lasso_coefficients.json'}")
    print(f"  {output_dir / 'covariate_predictive_metrics.json'}")
    print(f"  {output_dir / 'covariate_selection_summary.csv'}")
    print(f"  {output_dir / 'covariate_support.json'}")


if __name__ == "__main__":
    main()
