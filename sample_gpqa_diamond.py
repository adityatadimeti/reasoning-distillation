#!/usr/bin/env python3
"""
Sample a subset of a CSV file reproducibly and confirm reproducibility.
"""
import pandas as pd
import argparse
import sys

def sample_dataset(input_path: str, output_path: str, n: int, seed: int) -> None:
    df = pd.read_csv(input_path)
    sample_df = df.sample(n=n, random_state=seed)
    sample_df.to_csv(output_path, index=False)

    # confirm reproducibility
    df2 = pd.read_csv(input_path)
    sample_df2 = df2.sample(n=n, random_state=seed)
    if sample_df.equals(sample_df2):
        print(f"Reproducible sample confirmed: {n} rows saved to {output_path}")
    else:
        print("WARNING: Sample mismatch on rerun with same seed.", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Sample N rows from a CSV file reproducibly and confirm via rerun."
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Path to input CSV file"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Path to output CSV file"
    )
    parser.add_argument(
        "--n", "-n", type=int, default=50, help="Number of rows to sample"
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()
    sample_dataset(args.input, args.output, args.n, args.seed)


if __name__ == "__main__":
    main()
