import re
import time
import os
import requests
from io import StringIO
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


# =========================================
# API HELPERS
# =========================================
def extract_rest_tokens(html: str):
    tokens = re.findall(r"https://rest\.zse\.hr/web/([^/]+)/", html)
    return list(dict.fromkeys(tokens))


def fetch_html(session, url):
    r = session.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    r.raise_for_status()
    return r.text


def fetch_csv_with_token(session, token, isin, date_from, date_to, referer):
    url = (
        f"https://rest.zse.hr/web/{token}/security-history/XZAG/"
        f"{isin}/{date_from}/{date_to}/csv?language=HR"
    )
    r = session.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": referer,
            "X-Requested-With": "XMLHttpRequest",
        },
        timeout=60,
    )
    return r


def clean_df(df):
    if "Datum" in df.columns:
        df["Datum"] = pd.to_datetime(df["Datum"], errors="coerce")
        df = df.dropna(subset=["Datum"])
        df = df.sort_values("Datum")
    return df.reset_index(drop=True)


def zse_api_to_internal_df(df_raw):

    df = df_raw.copy()

    df = df.rename(columns={
        "Datum": "date",
        "Zadnja": "close",
        "Prva": "open",
        "Najviša": "high",
        "Najniža": "low",
        "Promet": "turnover",
    })

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for col in ["open","high","low","close","turnover"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date","close"])

    return df.sort_values("date").reset_index(drop=True)


# =========================================
# FEATURES & STRATEGY
# =========================================
def add_features(df):
    df["MA20"] = df["close"].rolling(20).mean()
    df["MA50"] = df["close"].rolling(50).mean()
    df["log_return"] = np.log(df["close"]).diff()
    return df


def rsi_ma_strategy(df):
    df["signal"] = 0
    buy = (df["MA20"] > df["MA50"])
    df.loc[buy, "signal"] = 1

    df["position"] = df["signal"].shift(1).fillna(0)
    df["ret"] = df["close"].pct_change().fillna(0)
    df["strategy_ret"] = df["position"] * df["ret"]
    df["cum_strategy"] = (1 + df["strategy_ret"]).cumprod()
    return df


def buy_and_hold(df):
    df["ret"] = df["close"].pct_change().fillna(0)
    df["cum_buyhold"] = (1 + df["ret"]).cumprod()
    return df


def arima_model_forecast(df):
    returns = df["log_return"].dropna()
    pval = adfuller(returns)[1]
    model = ARIMA(returns, order=(1,0,1)).fit()
    return pval


# =========================================
# HTML REPORT
# =========================================
def generate_html_report(results):
    html = f"""
    <html>
    <head>
        <title>Virtualna Burza</title>
        <style>
            body {{ font-family: Arial; padding: 40px; }}
            table {{ border-collapse: collapse; }}
            th, td {{ border: 1px solid #ccc; padding: 8px 12px; }}
            th {{ background-color: #f0f0f0; }}
        </style>
    </head>
    <body>
        <h1>Virtualna Burza Daily Report</h1>
        <p>Generated: {datetime.now()}</p>
        <table>
            <tr>
                <th>Ticker</th>
                <th>Buy & Hold</th>
                <th>RSI + MA</th>
                <th>ADF p-value</th>
            </tr>
    """

    for ticker, data in results.items():
        html += f"""
        <tr>
            <td>{ticker}</td>
            <td>{data['bh']:.2f}</td>
            <td>{data['rsi']:.2f}</td>
            <td>{data['pval']:.4f}</td>
        </tr>
        """

    html += "</table></body></html>"

    os.makedirs("output", exist_ok=True)
    with open("output/index.html", "w") as f:
        f.write(html)


# =========================================
# MAIN PIPELINE
# =========================================
if __name__ == "__main__":

    date_from = "2020-01-01"
    date_to = (date.today() - timedelta(days=1)).isoformat()

    stocks = {
        "ERNT": {"isin": "HRERNTRA0000", "page_url": "https://zse.hr/hr/papir/310?isin=HRERNTRA0000"},
        "HT":   {"isin": "HRHT00RA0005", "page_url": "https://zse.hr/hr/papir/310?isin=HRHT00RA0005"},
        "PODR": {"isin": "HRPODRRA0004", "page_url": "https://zse.hr/hr/papir/310?isin=HRPODRRA0004"},
    }

    results = {}

    with requests.Session() as session:
        for ticker, meta in stocks.items():
            html = fetch_html(session, meta["page_url"])
            tokens = extract_rest_tokens(html)

            for token in tokens:
                r = fetch_csv_with_token(
                    session, token,
                    meta["isin"],
                    date_from, date_to,
                    meta["page_url"]
                )

                if r.status_code == 200 and r.text.strip():
                    df_raw = pd.read_csv(StringIO(r.text), sep=";", decimal=",")
                    df_raw = clean_df(df_raw)

                    df = zse_api_to_internal_df(df_raw)
                    df = add_features(df)
                    df = buy_and_hold(df)
                    df = rsi_ma_strategy(df)

                    bh = df["cum_buyhold"].iloc[-1]
                    rsi = df["cum_strategy"].iloc[-1]
                    pval = arima_model_forecast(df)

                    results[ticker] = {
                        "bh": bh,
                        "rsi": rsi,
                        "pval": pval
                    }

                    break
            time.sleep(1)

    generate_html_report(results)
