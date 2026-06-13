#!/usr/bin/env python3
"""Build feature-by-stratum prevalence and win-rate tables for WIMHF runs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
WIMHF_REPO = PROJECT_DIR / "repos" / "wimhf"
if str(WIMHF_REPO) not in sys.path:
    sys.path.insert(0, str(WIMHF_REPO))

from wimhf.quickstart import (  # noqa: E402
    get_embeddings,
    load_and_preprocess_dataframe,
    load_config,
    train_sae,
)


DEFAULT_RUN_ROOT = (
    PROJECT_DIR
    / "outputs/reproduction/wimhf_exact/"
    / "wimhf_exact_full_prism_gpt5mini_s42_tron_20260608_6985771"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-context feature prevalence and feature-side win rates."
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
        help="Completed WIMHF run root containing configs/<dataset>_full.json.",
    )
    parser.add_argument(
        "--strata",
        nargs="+",
        default=["conversation_type"],
        help=(
            "Columns or derived strata to report. Supported derived strata: "
            "turn_bucket, subjectivity_pair, model_pair."
        ),
    )
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="val",
        help="Split used for reported prevalence/win-rate statistics.",
    )
    parser.add_argument(
        "--feature-set",
        choices=["kept", "global-top-k", "all"],
        default="kept",
        help="Which neurons to include in the table.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Global LASSO budget used when --feature-set=global-top-k.",
    )
    parser.add_argument(
        "--min-stratum-n",
        type=int,
        default=100,
        help="Drop strata with fewer binary-label examples than this.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to outputs/analysis/feature_strata/<run_name>.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_DIR / path


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def find_config(run_root: Path) -> Path:
    configs = sorted((run_root / "configs").glob("*.json"))
    if not configs:
        raise FileNotFoundError(f"No config JSON files found under {run_root / 'configs'}")
    if len(configs) > 1:
        print(f"[config] multiple configs found; using {configs[0]}")
    return configs[0]


def prepare_config(config_path: Path):
    cfg = load_config(str(config_path))
    cfg.dataset.path = str(resolve_path(Path(cfg.dataset.path)))
    if cfg.runtime.cache_dir is not None:
        cfg.runtime.cache_dir = resolve_path(Path(cfg.runtime.cache_dir))
    if cfg.runtime.checkpoint_dir is not None:
        cfg.runtime.checkpoint_dir = str(resolve_path(Path(cfg.runtime.checkpoint_dir)))
    cfg.runtime.retrain_sae = False
    return cfg


def dataset_dir_name(cfg) -> str:
    return str(cfg.dataset.name).lower().replace(" ", "_")


def find_result_dir(run_root: Path, cfg) -> Path:
    preferred = run_root / dataset_dir_name(cfg) / "results"
    if (preferred / "feature_table.jsonl").exists():
        return preferred
    matches = sorted(run_root.glob("*/results/feature_table.jsonl"))
    if len(matches) == 1:
        return matches[0].parent
    if not matches:
        raise FileNotFoundError(f"No feature_table.jsonl found under {run_root}")
    raise FileNotFoundError(
        f"Could not infer dataset results directory under {run_root}; matches: {matches}"
    )


def add_derived_strata(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "turn" in out.columns:
        turn = pd.to_numeric(out["turn"], errors="coerce")
        out["turn_bucket"] = np.select(
            [turn == 0, turn == 1, turn >= 2],
            ["turn_0", "turn_1", "turn_2_plus"],
            default="turn_unknown",
        )
    if "turn_id" in out.columns:
        turn_id = pd.to_numeric(out["turn_id"], errors="coerce")
        out["turn_bucket"] = np.select(
            [turn_id == 1, turn_id == 2, turn_id >= 3],
            ["turn_1", "turn_2", "turn_3_plus"],
            default="turn_unknown",
        )
    if {"subjectivity_A", "subjectivity_B"}.issubset(out.columns):
        out["subjectivity_pair"] = (
            out["subjectivity_A"].astype(str) + "_vs_" + out["subjectivity_B"].astype(str)
        )
    if {"model_A", "model_B"}.issubset(out.columns):
        pairs = [
            " | ".join(sorted([str(a), str(b)]))
            for a, b in zip(out["model_A"], out["model_B"])
        ]
        out["model_pair"] = pairs
    return out


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return center - margin, center + margin


def choose_neurons(
    feature_table: pd.DataFrame,
    lasso_path: Path,
    feature_set: str,
    top_k: int,
) -> list[int]:
    if feature_set == "all":
        return feature_table["neuron_idx"].astype(int).tolist()
    if feature_set == "kept":
        return feature_table.loc[feature_table["kept"], "neuron_idx"].astype(int).tolist()
    lasso = load_json(lasso_path)
    raw = lasso.get(str(top_k), lasso.get(top_k, {}))
    if not raw:
        raise KeyError(f"No top-{top_k} entry found in {lasso_path}")
    return [int(k) for k, _ in sorted(raw.items(), key=lambda item: abs(float(item[1])), reverse=True)]


def summarize_feature_strata(
    df: pd.DataFrame,
    activ: np.ndarray,
    label_col: str,
    feature_table: pd.DataFrame,
    neurons: list[int],
    strata: list[str],
    min_stratum_n: int,
) -> pd.DataFrame:
    meta = feature_table.set_index("neuron_idx").to_dict(orient="index")
    y = df[label_col].to_numpy()
    rows: list[dict[str, Any]] = []
    total_n = len(df)

    for stratum_col in strata:
        if stratum_col not in df.columns:
            raise KeyError(f"Stratum column not found after derivation: {stratum_col}")
        values = df[stratum_col].fillna("NA").astype(str).to_numpy()
        for value in sorted(pd.unique(values)):
            stratum_mask = values == value
            stratum_n = int(stratum_mask.sum())
            if stratum_n < min_stratum_n:
                continue
            stratum_y = y[stratum_mask]
            stratum_base_a_win_rate = float(np.mean(stratum_y == 1))

            for neuron_idx in neurons:
                z = activ[stratum_mask, neuron_idx]
                present = z != 0
                positive = z > 0
                negative = z < 0
                feature_side_wins = ((z > 0) & (stratum_y == 1)) | ((z < 0) & (stratum_y == 0))
                feature_side_n = int(present.sum())
                feature_wins = int(feature_side_wins.sum())
                feature_win_rate = (
                    float(feature_wins / feature_side_n) if feature_side_n else float("nan")
                )
                ci_low, ci_high = wilson_interval(feature_wins, feature_side_n)
                row_meta = meta.get(int(neuron_idx), {})
                rows.append(
                    {
                        "stratum_column": stratum_col,
                        "stratum_value": value,
                        "stratum_n": stratum_n,
                        "stratum_share": stratum_n / total_n,
                        "neuron_idx": int(neuron_idx),
                        "abbreviated_interpretation": row_meta.get("abbreviated_interpretation"),
                        "interpretation": row_meta.get("interpretation"),
                        "global_prevalence": row_meta.get("prevalence"),
                        "global_length_controlled_logit_coef": row_meta.get(
                            "length_controlled_logit_coef"
                        ),
                        "stratum_base_a_win_rate": stratum_base_a_win_rate,
                        "feature_prevalence": float(present.mean()),
                        "feature_side_n": feature_side_n,
                        "feature_win_rate": feature_win_rate,
                        "feature_win_rate_ci_low": ci_low,
                        "feature_win_rate_ci_high": ci_high,
                        "feature_win_rate_minus_stratum_base": (
                            feature_win_rate - stratum_base_a_win_rate
                            if feature_side_n
                            else float("nan")
                        ),
                        "positive_activation_prevalence": float(positive.mean()),
                        "negative_activation_prevalence": float(negative.mean()),
                        "positive_activation_a_win_rate": (
                            float(np.mean(stratum_y[positive] == 1))
                            if positive.any()
                            else float("nan")
                        ),
                        "negative_activation_b_win_rate": (
                            float(np.mean(stratum_y[negative] == 0))
                            if negative.any()
                            else float("nan")
                        ),
                        "mean_activation": float(np.mean(z)),
                        "mean_abs_activation": float(np.mean(np.abs(z))),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    run_root = resolve_path(args.run_root)
    if not run_root.exists():
        raise FileNotFoundError(f"Run root not found: {run_root}")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = PROJECT_DIR / "outputs" / "analysis" / "feature_strata" / run_root.name
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = find_config(run_root)
    cfg = prepare_config(config_path)
    result_dir = find_result_dir(run_root, cfg)
    feature_path = result_dir / "feature_table.jsonl"
    lasso_path = result_dir / "lasso_coefficients.json"
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature table not found: {feature_path}")
    if not lasso_path.exists():
        raise FileNotFoundError(f"LASSO coefficients not found: {lasso_path}")

    print(f"[config] {config_path}")
    print(f"[dataset] {cfg.dataset.name}")
    print(f"[output] {output_dir}")

    train_df, val_df, train_pred_df, val_pred_df, dedup_train_df, dedup_val_df = (
        load_and_preprocess_dataframe(cfg.dataset, cfg.runtime)
    )
    _, delta_train, delta_val, dedup_delta_train, dedup_delta_val = get_embeddings(
        train_df=train_df,
        val_df=val_df,
        dedup_train_df=dedup_train_df,
        dedup_val_df=dedup_val_df,
        dataset_cfg=cfg.dataset,
        embedding_cfg=cfg.embedding,
        cache_dir=cfg.runtime.cache_dir,
    )
    sae = train_sae(dedup_delta_train, dedup_delta_val, cfg.sae, cfg.runtime)

    if args.split == "train":
        pred_df = add_derived_strata(train_pred_df)
        pred_mask = train_df[cfg.dataset.label_column].isin([0, 1]).to_numpy()
        activ = sae.get_activations(delta_train, show_progress=False)[pred_mask]
    else:
        pred_df = add_derived_strata(val_pred_df)
        pred_mask = val_df[cfg.dataset.label_column].isin([0, 1]).to_numpy()
        activ = sae.get_activations(delta_val, show_progress=False)[pred_mask]

    feature_table = pd.read_json(feature_path, orient="records", lines=True)
    neurons = choose_neurons(feature_table, lasso_path, args.feature_set, args.top_k)
    table = summarize_feature_strata(
        df=pred_df,
        activ=activ,
        label_col=cfg.dataset.label_column,
        feature_table=feature_table,
        neurons=neurons,
        strata=args.strata,
        min_stratum_n=args.min_stratum_n,
    )

    suffix = f"{args.split}_{args.feature_set}"
    if args.feature_set == "global-top-k":
        suffix += f"{args.top_k}"
    csv_path = output_dir / f"feature_strata_{suffix}.csv"
    table.to_csv(csv_path, index=False)

    support = (
        table[["stratum_column", "stratum_value", "stratum_n", "stratum_share"]]
        .drop_duplicates()
        .sort_values(["stratum_column", "stratum_value"])
    )
    support_path = output_dir / f"strata_support_{suffix}.csv"
    support.to_csv(support_path, index=False)

    payload = {
        "run_root": str(run_root),
        "config": str(config_path),
        "dataset": cfg.dataset.name,
        "split": args.split,
        "feature_set": args.feature_set,
        "top_k": args.top_k if args.feature_set == "global-top-k" else None,
        "strata": args.strata,
        "min_stratum_n": args.min_stratum_n,
        "n_rows": int(len(table)),
        "n_neurons": int(len(neurons)),
        "neurons": neurons,
        "outputs": {
            "feature_strata_csv": str(csv_path),
            "strata_support_csv": str(support_path),
        },
    }
    dump_json(output_dir / f"feature_strata_{suffix}.json", payload)

    print("[done] wrote:")
    print(f"  {csv_path}")
    print(f"  {support_path}")
    print(f"  {output_dir / f'feature_strata_{suffix}.json'}")


if __name__ == "__main__":
    main()
