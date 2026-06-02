"""Download historical index data used by the model.

Replaces the original Quandl-based fetcher (the Quandl free tier shut down
in 2018) with `yfinance`, which still mirrors Yahoo Finance.

Usage:
    python download_data.py                   # default range, refreshes data/
    python download_data.py --start 2015-01-01
    python download_data.py --out custom_dir/
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import yfinance as yf


INDICES = {
    "^AORD": "All Ordinaries (Australia)",
    "^N225": "Nikkei 225 (Japan)",
    "^HSI": "Hang Seng (Hong Kong)",
    "^GDAXI": "DAX (Germany)",
    "^NYA": "NYSE Composite (US)",
    "^DJI": "Dow Jones Industrial Average (US)",
    "^GSPC": "S&P 500 (US)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2012-05-12", help="ISO start date")
    parser.add_argument("--end", default=date.today().isoformat(), help="ISO end date")
    parser.add_argument("--out", default="data", help="Output directory")
    return parser.parse_args()


def download(ticker: str, start: str, end: str, out: Path) -> Path:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    path = out / f"{ticker}.csv"
    df.to_csv(path)
    return path


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for ticker, label in INDICES.items():
        path = download(ticker, args.start, args.end, out)
        print(f"  {ticker:8s}  {label:35s}  -> {path}")


if __name__ == "__main__":
    main()
