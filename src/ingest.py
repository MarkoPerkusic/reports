import re
import time
import requests
import pandas as pd
import numpy as np
from io import StringIO
from datetime import date, timedelta


# =========================================
# DUMMY (za testiranje bez API-ja)
# =========================================
def load_dummy_data() -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=200, freq="D")
    price = 100 + np.cumsum(np.random.normal(0, 1, len(dates)))
    df = pd.DataFrame({
        "date": dates,
        "close": price
    })
    df["open"] = df["close"] + np.random.normal(0, 0.5, len(df))
    df["high"] = df[["open", "close"]].max(axis=1) + np.abs(np.random.normal(0, 1, len(df)))
    df["low"] = df[["open", "close"]].min(axis=1) - np.abs(np.random.normal(0, 1, len(df)))
    df["volume"] = np.random.randint(1000, 10000, len(df))
    df = df[["date", "open", "high", "low", "close", "volume"]]
    return df


# =========================================
# ZSE API HELPERS
# =========================================
def _extract_rest_tokens(html: str) -> list[str]:
    tokens = re.findall(r"https://rest\.zse\.hr/web/([^/]+)/", html)
    return list(dict.fromkeys(tokens))


def _fetch_html(session, url: str) -> str:
    r = session.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    r.raise_for_status()
    return r.text


def _fetch_csv(session, token: str, isin: str, date_from: str, date_to: str, referer: str):
    url = (
        f"https://rest.zse.hr/web/{token}/security-history/XZAG/"
        f"{isin}/{date_from}/{date_to}/csv?language=HR"
    )
    return session.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": referer,
            "X-Requested-With": "XMLHttpRequest",
        },
        timeout=60,
    )


def _parse_and_normalize(raw_csv: str) -> pd.DataFrame:
    df = pd.read_csv(StringIO(raw_csv), sep=";", decimal=",")

    # mapiranje stupaca
    df = df.rename(columns={
        "last_price":  "close",
        "open_price":  "open",
        "high_price":  "high",
        "low_price":   "low",
    })

    # parsiranje datuma
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # očisti i konvertiraj numeričke stupce
    numeric_cols = ["open", "high", "low", "close", "volume", "turnover", "vwap_price"]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .replace({"-": None, "": None})
                .str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # makni nevalidne datume
    df = df.dropna(subset=["date"])

    # fallback: koristi vwap ako nema close
    if "close" in df.columns and "vwap_price" in df.columns:
        df["close"] = df["close"].fillna(df["vwap_price"])

    # makni retke gdje OHLC nije validan
    required_cols = ["open", "high", "low", "close"]
    existing = [c for c in required_cols if c in df.columns]

    if existing:
        df = df.dropna(subset=existing)

    # ako nema ništa → vrati prazan df
    if df.empty:
        return pd.DataFrame()

    # sortiranje
    df = df.sort_values("date").reset_index(drop=True)

    return df


# =========================================
# GLAVNA FUNKCIJA
# =========================================
STOCKS = {
    "ERNT": {"isin": "HRERNTRA0000", "page_url": "https://zse.hr/hr/papir/310?isin=HRERNTRA0000"},
    "HT":   {"isin": "HRHT00RA0005", "page_url": "https://zse.hr/hr/papir/310?isin=HRHT00RA0005"},
    "PODR": {"isin": "HRPODRRA0004", "page_url": "https://zse.hr/hr/papir/310?isin=HRPODRRA0004"},
}


def load_data(
    stocks: dict = STOCKS,
    date_from: str = "2023-01-01",
    date_to: str | None = None,
    sleep_between: float = 1.0,
) -> dict[str, pd.DataFrame]:
    """
    Dohvaća podatke s ZSE API-ja za sve dionice.

    Vraća dict: { "ERNT": df, "HT": df, "PODR": df }

    Ako dohvat za određenu dionicu ne uspije, taj ticker se preskače.
    """
    if date_to is None:
        date_to = (date.today() - timedelta(days=1)).isoformat()

    results = {}

    with requests.Session() as session:
        for ticker, meta in stocks.items():
            try:
                html = _fetch_html(session, meta["page_url"])
                tokens = _extract_rest_tokens(html)

                for token in tokens:
                    r = _fetch_csv(session, token, meta["isin"], date_from, date_to, meta["page_url"])

                    if r.status_code == 200 and r.text.strip():
                        df = _parse_and_normalize(r.text)
                        results[ticker] = df
                        print(f"[ingest] {ticker}: {len(df)} redaka dohvaćeno")
                        break
                else:
                    print(f"[ingest] {ticker}: nije pronađen valjan token, preskačem")

            except Exception as e:
                print(f"[ingest] {ticker}: greška — {e}")

            time.sleep(sleep_between)

    return results
