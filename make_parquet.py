"""
Combine all order-book CSVs into a single snappy-compressed parquet file.

Usage:
    python make_parquet.py
    python make_parquet.py --out data/orderbook/btcusd_full.parquet
"""
import argparse
from pathlib import Path
import polars as pl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv-dir", default="data/csv",
        help="Directory containing order-book-*.csv files (default: data/csv)",
    )
    parser.add_argument(
        "--out", default="data/orderbook/btcusd_full.parquet",
        help="Output parquet path (default: data/orderbook/btcusd_full.parquet)",
    )
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(csv_dir.glob("order-book-*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No order-book-*.csv files found in {csv_dir}")

    print(f"Found {len(csv_files)} CSV files")

    frames = []
    total_rows = 0
    for f in csv_files:
        df = pl.read_csv(f, try_parse_dates=True)
        frames.append(df)
        total_rows += len(df)
        print(f"  {f.name}: {len(df):,} rows")

    print(f"\nCombining {total_rows:,} total rows...")
    combined = pl.concat(frames, how="vertical")

    # Normalise column name so downstream pipeline can use it directly
    if "mid" in combined.columns and "mid_price" not in combined.columns:
        combined = combined.rename({"mid": "mid_price"})

    # Sort chronologically
    if "ts" in combined.columns:
        combined = combined.sort("ts")

    combined.write_parquet(out_path, compression="snappy")

    size_mb = out_path.stat().st_size / (1024 ** 2)
    print(f"\nSaved → {out_path}  ({size_mb:.1f} MB,  {len(combined):,} rows)")


if __name__ == "__main__":
    main()
