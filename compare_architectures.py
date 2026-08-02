"""
Compare results across all 10 DTI architectures on a given dataset.

Usage:
    python compare_architectures.py --dataset humans
    python compare_architectures.py --dataset bindingdb --metric pr_auc
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_results(arch_name: str, dataset: str) -> dict:
    """Load results.json for a given architecture and dataset."""
    results_path = Path("architectures") / arch_name / "results" / dataset / "results.json"
    if not results_path.exists():
        return None
    try:
        with open(results_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"  Error loading {arch_name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Compare results across all DTI architectures."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., 'humans', 'bindingdb', 'biosnap', 'celegans')",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="pr_auc",
        help="Metric to sort by (default: pr_auc)",
    )

    args = parser.parse_args()

    # List of all 10 architectures (placeholder for now)
    architectures = [
        "mamba_mlp_fp",
        "bilstm_mlp",
        "mamba_attentionpool_mlp",
        "mamba_cross_mlp",
        "meanpool_mlp",
        # Architectures 6-10 (to be added)
        # "arch_6",
        # "arch_7",
        # "arch_8",
        # "arch_9",
        # "arch_10",
    ]

    print(f"\n{'='*100}")
    print(f"  Architecture Comparison: {args.dataset.upper()} Dataset")
    print(f"{'='*100}\n")

    # Load all results
    all_results = {}
    for arch in architectures:
        print(f"  Loading {arch}...", end=" ")
        results = load_results(arch, args.dataset)
        if results:
            all_results[arch] = results
            print("✓")
        else:
            print("✗ (not found)")

    if not all_results:
        print("\n  No results found. Run training first with:")
        print(f"  python architectures/ARCH/train.py --dataset {args.dataset}")
        return

    # Compile comparison table
    comparison_data = []

    for arch, results in all_results.items():
        row = {"Architecture": arch}
        metrics = results.get("metrics", {})

        for metric_name, metric_data in metrics.items():
            val_data = metric_data.get("val", {})
            test_data = metric_data.get("test", {})

            val_mean = val_data.get("mean")
            test_mean = test_data.get("mean")
            test_std = test_data.get("std")

            # Add to row
            row[f"{metric_name}_val"] = val_mean
            row[f"{metric_name}_test"] = test_mean
            row[f"{metric_name}_test_std"] = test_std

        comparison_data.append(row)

    # Create DataFrame
    comparison_df = pd.DataFrame(comparison_data)

    # Sort by test metric
    sort_col = f"{args.metric}_test"
    if sort_col in comparison_df.columns:
        comparison_df = comparison_df.sort_values(sort_col, ascending=False)

    # Display results
    print(f"\n  Sorted by Test {args.metric.upper()}\n")
    print(f"  {'Architecture':<25}  {'Val PR-AUC':>12}  {'Test PR-AUC':>12}  {'Test ROC-AUC':>12}")
    print(f"  {'─'*25}  {'─'*12}  {'─'*12}  {'─'*12}")

    for _, row in comparison_df.iterrows():
        arch = row["Architecture"]
        val_pr = row.get("pr_auc_val", np.nan)
        test_pr = row.get("pr_auc_test", np.nan)
        test_roc = row.get("roc_auc_test", np.nan)

        val_str = f"{val_pr:.4f}" if not np.isnan(val_pr) else "N/A"
        test_pr_str = f"{test_pr:.4f}" if not np.isnan(test_pr) else "N/A"
        test_roc_str = f"{test_roc:.4f}" if not np.isnan(test_roc) else "N/A"

        print(
            f"  {arch:<25}  {val_str:>12}  {test_pr_str:>12}  {test_roc_str:>12}"
        )

    print(f"  {'─'*25}  {'─'*12}  {'─'*12}  {'─'*12}\n")

    # Save comparison to CSV
    output_path = Path(f"comparison_{args.dataset}.csv")
    comparison_df.to_csv(output_path, index=False)
    print(f"  ✓ Saved comparison to {output_path}\n")


if __name__ == "__main__":
    main()
