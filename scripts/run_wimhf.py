#!/usr/bin/env python3
"""Run project WIMHF reproductions with generated per-run configs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
WIMHF_REPO = PROJECT_DIR / "repos" / "wimhf"
if str(WIMHF_REPO) not in sys.path:
    sys.path.insert(0, str(WIMHF_REPO))

from wimhf.quickstart import load_config, run_wimhf_pipeline  # noqa: E402


BASE_CONFIGS = {
    "community_align": PROJECT_DIR / "configs/community_align_wimhf_local.json",
    "prism": PROJECT_DIR / "configs/prism_wimhf_local.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run WIMHF on the project preference datasets."
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        choices=sorted(BASE_CONFIGS),
        default=["community_align", "prism"],
        help="Dataset keys to run. Defaults to both target datasets.",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "smoke"],
        default=os.environ.get("RUN_MODE", "full"),
        help="Run full by default; smoke is a quick sanity check.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Directory for generated configs, logs, and results.",
    )
    parser.add_argument(
        "--local-model",
        default=os.environ.get(
            "LOCAL_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507"
        ),
        help="Local vLLM model used for annotation/scoring.",
    )
    parser.add_argument(
        "--local-interpreter",
        action="store_true",
        default=os.environ.get("LOCAL_INTERPRETER", "0") == "1",
        help="Also use the local model for feature interpretation.",
    )
    parser.add_argument(
        "--smoke-rows",
        type=int,
        default=int(os.environ.get("SMOKE_ROWS", "512")),
        help="Rows per dataset in smoke mode.",
    )
    return parser.parse_args()


def ensure_openai_key() -> None:
    if not os.environ.get("OAI_WIMHF") and os.environ.get("OPENAI_API_KEY"):
        os.environ["OAI_WIMHF"] = os.environ["OPENAI_API_KEY"]
    if not os.environ.get("OAI_WIMHF"):
        raise RuntimeError(
            "OAI_WIMHF or OPENAI_API_KEY must be set for embeddings and abbreviations."
        )


def dataset_tag(datasets: list[str]) -> str:
    return "ca_prism" if datasets == ["community_align", "prism"] else "_".join(datasets)


def default_run_root(mode: str, datasets: list[str]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"wimhf_local_{mode}_{dataset_tag(datasets)}_s42_{timestamp}"
    return PROJECT_DIR / "outputs" / "reproduction" / "wimhf_local" / run_name


def read_base_config(dataset: str) -> dict[str, Any]:
    with BASE_CONFIGS[dataset].open() as f:
        return json.load(f)


def write_smoke_sample(
    cfg: dict[str, Any],
    dataset: str,
    run_root: Path,
    smoke_rows: int,
) -> None:
    source_path = PROJECT_DIR / cfg["dataset"]["path"]
    sample_path = run_root / "data" / f"{dataset}_smoke.jsonl"
    sample_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_json(source_path, orient="records", lines=True)
    n_rows = min(len(df), smoke_rows)
    df = df.sample(n=n_rows, random_state=cfg["dataset"].get("random_seed", 42))
    df.reset_index(drop=True).to_json(sample_path, orient="records", lines=True)
    cfg["dataset"]["path"] = str(sample_path)


def make_run_config(
    dataset: str,
    mode: str,
    run_root: Path,
    local_model: str,
    local_interpreter: bool,
    smoke_rows: int,
) -> Path:
    cfg = read_base_config(dataset)

    if local_interpreter:
        cfg["interpretation"]["interpreter_model"] = local_model
        cfg["interpretation"]["completion_kwargs"] = {}
    cfg["interpretation"]["annotator_model"] = local_model
    cfg["interpretation"]["abbreviator_model"] = "gpt-5-mini"
    cfg["runtime"]["checkpoint_dir"] = str(run_root / dataset / "checkpoints")
    cfg["runtime"]["cache_dir"] = str(run_root / dataset / "cache")
    cfg["runtime"]["retrain_sae"] = True

    if mode == "smoke":
        write_smoke_sample(cfg, dataset, run_root, smoke_rows)
        cfg["dataset"]["train_split_size"] = 0.8
        cfg["sae"].update(
            {
                "M": 8,
                "K": 2,
                "prefix_lengths": [4, 8],
                "batch_size": 64,
                "n_epochs": 3,
                "min_epochs": 1,
                "patience": 1,
            }
        )
        cfg["interpretation"].update(
            {
                "n_candidates": 1,
                "interpret_n_examples": 3,
                "scoring_n_examples": 12,
                "n_workers_interpretation": 1,
                "n_workers_annotation": 1,
                "max_interpretation_tokens": 512,
                "p_value_threshold": None,
            }
        )
        cfg["selection"]["lasso_top_k"] = [3]

    config_dir = run_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{dataset}_{mode}.json"
    with config_path.open("w") as f:
        json.dump(cfg, f, indent=2)
    return config_path


def write_wimhf_outputs(results: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_table: pd.DataFrame = results["feature_table"]
    feature_table.to_json(output_dir / "feature_table.jsonl", orient="records", lines=True)

    outputs = {
        "summary_metrics.json": results["summary_metrics"],
        "predictive_metrics.json": results["auc_metrics"],
        "lasso_coefficients.json": results["lasso_coef_maps"],
        "logit_coefficients.json": results["logit_coef_map"],
    }
    for filename, payload in outputs.items():
        with (output_dir / filename).open("w") as f:
            json.dump(payload, f, indent=2)


def run_dataset(
    dataset: str,
    mode: str,
    run_root: Path,
    local_model: str,
    local_interpreter: bool,
    smoke_rows: int,
) -> None:
    config_path = make_run_config(
        dataset=dataset,
        mode=mode,
        run_root=run_root,
        local_model=local_model,
        local_interpreter=local_interpreter,
        smoke_rows=smoke_rows,
    )
    output_dir = run_root / dataset / "results"

    print(f"Starting {dataset}: {config_path}", flush=True)
    cfg = load_config(str(config_path))
    results = run_wimhf_pipeline(cfg)
    write_wimhf_outputs(results, output_dir)
    print(f"Finished {dataset}: {output_dir}", flush=True)


def main() -> None:
    args = parse_args()
    ensure_openai_key()

    run_root = args.run_root or default_run_root(args.mode, args.datasets)
    run_root.mkdir(parents=True, exist_ok=True)

    print(f"Run root: {run_root}", flush=True)
    print(f"Mode: {args.mode}", flush=True)
    print(f"Local model: {args.local_model}", flush=True)
    print(f"Local interpreter: {int(args.local_interpreter)}", flush=True)
    print(f"Datasets: {' '.join(args.datasets)}", flush=True)

    for dataset in args.datasets:
        run_dataset(
            dataset=dataset,
            mode=args.mode,
            run_root=run_root,
            local_model=args.local_model,
            local_interpreter=args.local_interpreter,
            smoke_rows=args.smoke_rows,
        )

    print(f"All requested WIMHF runs completed under {run_root}", flush=True)


if __name__ == "__main__":
    main()
