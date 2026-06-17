import re
import os
import time
import requests
import pandas as pd
import numpy as np
from io import StringIO
from datetime import date, timedelta

from src.config import PAW_BASE_URL, STOCKS, USE_DUMMY_DATA


START_DATE = "2023-01-01"
PAW_API_KEY = os.getenv("PAW_API_KEY")

def load_dummy_data() -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=200, freq="D")
    price = 100 + np.cumsum(np.random.normal(0, 1, len(dates)))

    df = pd.DataFrame({
        "date": dates,
        "close": price,
    })

    df["open"] = df["close"] + np.random.normal(0, 0.5, len(df))
    df["high"] = df[["open", "close"]].max(axis=1) + abs(np.random.normal(0, 1, len(df)))
    df["low"] = df[["open", "close"]].min(axis=1) - abs(np.random.normal(0, 1, len(df)))
    df["volume"] = np.random.randint(1000, 10000, len(df))

    return df[["date", "open", "high", "low", "close", "volume"]]


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


def _parse_for_paw(raw_csv: str) -> pd.DataFrame:
    df = pd.read_csv(StringIO(raw_csv), sep=";", decimal=",")

    df = df.rename(columns={
        "trading_model_id": "trading_model",
        "change_prev_close_percentage": "change_pct",
    })

    expected_cols = [
        "isin", "symbol", "mic", "date", "trading_model",
        "open_price", "high_price", "low_price", "last_price", "vwap_price",
        "change_pct", "num_trades", "volume", "turnover",
        "price_currency", "turnover_currency",
    ]

    df = df[expected_cols]

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["isin", "date", "trading_model"])

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.astype(object).where(pd.notnull(df), None)

    return df


def _normalize_for_strategy(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    df = df.copy()

    if "trading_model" in df.columns:
        df = df[df["trading_model"] == "CT"]

    df = df.rename(columns={
        "last_price": "close",
        "open_price": "open",
        "high_price": "high",
        "low_price": "low",
    })

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    numeric_cols = [
        "open", "high", "low", "close",
        "volume", "turnover", "vwap_price",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date"])

    if "close" in df.columns and "vwap_price" in df.columns:
        df["close"] = df["close"].fillna(df["vwap_price"])

    df = df.dropna(subset=["open", "high", "low", "close"])

    return df.sort_values("date").reset_index(drop=True)


def _get_last_date_from_paw(isin: str) -> str | None:
    url = f"{PAW_BASE_URL}/lastdate/{isin}"

    r = requests.get(url, timeout=60)
    r.raise_for_status()

    return r.json().get("last_date")


def _post_prices_to_paw(df: pd.DataFrame) -> None:
    if df.empty:
        print("[paw] nema novih redaka za spremanje")
        return

    url = f"{PAW_BASE_URL}/prices"
    payload = df.to_dict(orient="records")

    r = requests.post(
        url,
        json=payload,
        headers={
            "X-API-Key": PAW_API_KEY
        },
        timeout=60,
    )

    if not r.ok:
        print("[paw] POST error status:", r.status_code)
        print("[paw] POST error body:", r.text)
        r.raise_for_status()

    print(f"[paw] spremljeno redaka: {len(payload)}")


def _get_all_prices_from_paw(isin: str) -> pd.DataFrame:
    url = f"{PAW_BASE_URL}/prices/{isin}"

    r = requests.get(url, timeout=60)

    if r.status_code == 404:
        return pd.DataFrame()

    r.raise_for_status()

    return pd.DataFrame(r.json())


def _calculate_date_from(last_date: str | None) -> str:
    if last_date is None:
        return START_DATE

    return (
        pd.to_datetime(last_date) + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")


def load_data(
    stocks: dict = STOCKS,
    date_from: str | None = None,
    date_to: str | None = None,
    sleep_between: float = 1.0,
) -> dict[str, pd.DataFrame]:

    if USE_DUMMY_DATA:
        return {
            "DUMMY": load_dummy_data()
        }

    default_date_to = (date.today() - timedelta(days=1)).isoformat()
    manual_range = date_from is not None

    if date_to is None:
        date_to = default_date_to

    results = {}

    with requests.Session() as session:
        for ticker, meta in stocks.items():
            isin = meta["isin"]

            try:
                last_date = _get_last_date_from_paw(isin)

                if manual_range:
                    fetch_from = date_from
                else:
                    fetch_from = _calculate_date_from(last_date)

                print(
                    f"[ingest] {ticker}: "
                    f"PAW last_date={last_date}, "
                    f"fetch od {fetch_from} do {date_to}"
                )

                if fetch_from <= date_to:
                    html = _fetch_html(session, meta["page_url"])
                    tokens = _extract_rest_tokens(html)

                    fetched_new = False

                    for token in tokens:
                        r = _fetch_csv(
                            session=session,
                            token=token,
                            isin=isin,
                            date_from=fetch_from,
                            date_to=date_to,
                            referer=meta["page_url"],
                        )

                        if r.status_code == 200 and r.text.strip():
                            df_new = _parse_for_paw(r.text)
                            _post_prices_to_paw(df_new)
                            fetched_new = True
                            break

                    if not fetched_new:
                        print(f"[ingest] {ticker}: nema novih podataka iz ZSE")
                else:
                    print(f"[ingest] {ticker}: PAW je već ažuran")

                df_all = _get_all_prices_from_paw(isin)
                df_strategy = _normalize_for_strategy(df_all)

                if not df_strategy.empty:
                    results[ticker] = df_strategy
                    print(f"[ingest] {ticker}: {len(df_strategy)} redaka učitano iz PAW")
                else:
                    print(f"[ingest] {ticker}: nema validnih podataka nakon PAW učitavanja")

            except Exception as e:
                print(f"[ingest] {ticker}: greška — {e}")

            time.sleep(sleep_between)

    return results