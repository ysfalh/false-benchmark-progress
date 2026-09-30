"""Regenerate the reported tables from all release trial records."""

import argparse
from pathlib import Path
import pandas as pd
from benchmark_progress.benchmarks import ROOT
from benchmark_progress.reporting import write_tables

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, default=ROOT/'results/release_v2')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    write_tables(pd.read_parquet(args.release/'trials.parquet'), args.output)
    print(f'Wrote the complete tables to {args.output}')
