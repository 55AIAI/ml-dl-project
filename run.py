"""Reproducible classical benchmark; run from the repository root."""
import argparse
import json
from pathlib import Path

import pandas as pd

from ml_lab.experiments import benchmark, load_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["digits", "diabetes", "synthetic"], default="digits")
    parser.add_argument("--csv", type=Path, help="Optional Diabetes CSV with exact documented columns")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="At most 400 rows, 3-fold CV, one config/model")
    parser.add_argument("--output", type=Path, default=Path("output/benchmark"))
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be an integer in [0, 2**32)")
    if args.csv and args.dataset != "diabetes":
        parser.error("--csv is supported only with --dataset diabetes")
    if args.output.exists():
        parser.error("output directory already exists; choose a new path")
    try:
        data = load_dataset(args.dataset, args.csv, args.quick, args.seed)
        result = benchmark(data, args.seed, args.quick)
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                                                encoding="utf-8")
        pd.DataFrame(result["candidates"]).to_csv(args.output / "development.csv", index=False)
        text = (f"# {data.name}: {result['mode']} run\n\n"
                f"Selected on validation weighted F1: **{result['selected_model']}**.\n\n"
                f"Final held-out accuracy: {result['test']['accuracy']:.4f}; "
                f"weighted F1: {result['test']['f1_weighted']:.4f}.\n\n"
                f"Most-frequent baseline accuracy: {result['baseline_test']['accuracy']:.4f}.\n\n"
                "Imputation/scaling run inside each CV fold. The selected pipeline is refitted "
                "on train + validation before the single test evaluation. Candidate CV scores "
                "are tuning scores, not an independent generalization estimate.\n")
        (args.output / "README.md").write_text(text, encoding="utf-8")
        print(text)
        print(f"Saved report to {args.output}")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
