import pandas as pd
import yfinance as yf
import time

RETURNS_PATH = "data/returns.csv"
OUT_PATH = "data/metadata.csv"

# (yfinance name, full display name used by modules, short label used in charts)
SECTOR_NAMES = [
    ("Technology",             "Technology",    "Tech"),
    ("Financial Services",     "Financials",    "Fin"),
    ("Healthcare",             "Healthcare",    "Health"),
    ("Consumer Cyclical",      "Cons. Discr.",  "Discr"),
    ("Consumer Defensive",     "Cons. Staples", "Staples"),
    ("Industrials",            "Industrials",   "Indust"),
    ("Energy",                 "Energy",        "Energy"),
    ("Communication Services", "Communication", "Comm"),
    ("Utilities",              "Utilities",     "Util"),
    ("Real Estate",            "Real Estate",   "RE"),
    ("Basic Materials",        "Materials",     "Mat"),
]

SECTOR_FULL = {yf_name: full for yf_name, full, _    in SECTOR_NAMES}
SECTOR_ABBR = {yf_name: abbr for yf_name, _,    abbr in SECTOR_NAMES}


def fetch_metadata(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for i, ticker in enumerate(tickers):
        print(f"  [{i+1}/{len(tickers)}] {ticker} ...", end=" ", flush=True)
        try:
            info = yf.Ticker(ticker).info
            raw  = info.get("sector", "Other")
            print(SECTOR_FULL.get(raw, raw))
            rows.append({
                "ticker":      ticker,
                "sector":      SECTOR_FULL.get(raw, raw),
                "sector_abbr": SECTOR_ABBR.get(raw, raw),
                "industry":    info.get("industry", ""),
                "name":        info.get("shortName", ticker),
            })
        except Exception as e:
            print(f"ERROR ({e})")
            rows.append({
                "ticker": ticker, "sector": "Other", "sector_abbr": "Other",
                "industry": "", "name": ticker,
            })
        time.sleep(0.2)
    return pd.DataFrame(rows).set_index("ticker")


if __name__ == "__main__":
    tickers = list(pd.read_csv(RETURNS_PATH, index_col=0, nrows=1).columns)
    print(f"Fetching metadata for {len(tickers)} tickers...")
    df = fetch_metadata(tickers)
    df.to_csv(OUT_PATH)
    print(f"\nSaved {OUT_PATH}")
    print(df["sector"].value_counts().to_string())
