import json
import re
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")
OUT_DIR = DATA_DIR / "csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N = 5

def parse_file(path: Path) -> pd.DataFrame:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            data = obj.get("data", {})
            records.append({
                "ts_ms": obj.get("ts"),
                "type":  obj.get("type"),
                "bids":  data.get("b", []),
                "asks":  data.get("a", []),
            })
    raw = pd.DataFrame(records)
    raw["ts"] = pd.to_datetime(raw["ts_ms"], unit="ms", utc=True)
    return raw


def reconstruct_book(records: pd.DataFrame, top_n: int = TOP_N) -> pd.DataFrame:
    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    rows = []

    for rec in records.itertuples():
        for value in rec.bids:
            p, q = float(value[0]), float(value[1])
            if q == 0:
                bids.pop(p, None)
            else:
                bids[p] = q

        for value in rec.asks:
            p, q = float(value[0]), float(value[1])
            if q == 0:
                asks.pop(p, None)
            else:
                asks[p] = q

        if not bids or not asks:
            continue

        sorted_bids = sorted(bids.items(), reverse=True)[:top_n]
        sorted_asks = sorted(asks.items())[:top_n]
        best_bid, best_ask = sorted_bids[0][0], sorted_asks[0][0]

        row = {
            "ts":     rec.ts,
            "type":   rec.type,
            "mid":    (best_bid + best_ask) / 2,
            "spread": best_ask - best_bid,
        }
        for i, (p, q) in enumerate(sorted_bids):
            row[f"bid_p{i}"] = p
            row[f"bid_q{i}"] = q
        for i, (p, q) in enumerate(sorted_asks):
            row[f"ask_p{i}"] = p
            row[f"ask_q{i}"] = q
        rows.append(row)

    return pd.DataFrame(rows)


def date_from_filename(path: Path) -> str:
    """Extract YYYY-MM-DD from filenames like 2026-03-26_BTCUSD_ob200.data"""
    match = re.match(r"(\d{4}-\d{2}-\d{2})", path.name)
    if not match:
        raise ValueError(f"Cannot parse date from filename: {path.name}")
    return match.group(1)


def main():
    data_files = sorted(DATA_DIR.glob("*.data"))
    print(f"Found {len(data_files)} .data files")

    for path in data_files:
        date_str = date_from_filename(path)
        out_path = OUT_DIR / f"order-book-{date_str}.csv"

        if out_path.exists():
            print(f"  SKIP  {out_path.name} (already exists)")
            continue

        print(f"  Processing {path.name} ...", end=" ", flush=True)
        raw = parse_file(path)
        book = reconstruct_book(raw, top_n=TOP_N)
        book.to_csv(out_path, index=False)
        print(f"-> {len(book):,} rows  saved to {out_path.name}")

    print("Done.")


if __name__ == "__main__":
    main()
