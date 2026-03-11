import os
import re
import time
import requests
from io import StringIO
from datetime import date, timedelta, datetime

import pandas as pd
import numpy as np

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

import plotly.graph_objects as go
from plotly.offline import plot


# =====================================================
# API HELPERS
# =====================================================

def fetch_html(session, url):
    r = session.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    r.raise_for_status()
    return r.text


def extract_rest_tokens(html):
    tokens = re.findall(r"https://rest\.zse\.hr/web/([^/]+)/", html)
    return list(dict.fromkeys(tokens))


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


# =====================================================
# DATA CLEANING
# =====================================================

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


# =====================================================
# FEATURES
# =====================================================

def add_features(df):

    df["MA20"] = df["close"].rolling(20).mean()
    df["MA50"] = df["close"].rolling(50).mean()

    df["log_return"] = np.log(df["close"]).diff()

    df["ret"] = df["close"].pct_change().fillna(0)

    df["cum_buyhold"] = (1 + df["ret"]).cumprod()

    return df


# =====================================================
# STRATEGY
# =====================================================

def run_strategy(df):

    df["signal"] = 0

    df.loc[df["MA20"] > df["MA50"], "signal"] = 1

    df["position"] = df["signal"].shift(1).fillna(0)

    df["strategy_ret"] = df["position"] * df["ret"]

    df["cum_strategy"] = (1 + df["strategy_ret"]).cumprod()

    return df


def get_current_signal(df):

    last = df.iloc[-1]

    if last["MA20"] > last["MA50"]:
        return "BUY"
    else:
        return "CASH"


# =====================================================
# ARIMA TEST
# =====================================================

def arima_test(df):

    returns = df["log_return"].dropna()

    pval = adfuller(returns)[1]

    model = ARIMA(returns, order=(1,0,1)).fit()

    return pval


# =====================================================
# CHARTS
# =====================================================

def generate_price_chart(df, ticker):

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["close"],
        name="Price"
    ))

    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["MA20"],
        name="MA20"
    ))

    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["MA50"],
        name="MA50"
    ))

    fig.update_layout(
        title=f"{ticker} Price",
        template="plotly_white"
    )

    return plot(fig, include_plotlyjs=False, output_type="div")


def generate_equity_chart(df, ticker):

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["cum_strategy"],
        name="Strategy"
    ))

    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["cum_buyhold"],
        name="Buy & Hold"
    ))

    fig.update_layout(
        title=f"{ticker} Strategy vs BuyHold",
        template="plotly_white"
    )

    return plot(fig, include_plotlyjs=False, output_type="div")


# =====================================================
# HTML REPORT
# =====================================================

def generate_html(summary_rows, charts):

    top_stock = max(summary_rows, key=lambda x: x["strategy"])

    portfolio_return = sum(r["strategy"] for r in summary_rows) / len(summary_rows)

    wins = sum(1 for r in summary_rows if r["strategy"] > r["buyhold"])

    win_rate = wins / len(summary_rows)

    html = f"""
<html>
<head>

<title>Virtualna Burza</title>

<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>

<style>

body {{
font-family: Arial;
margin:40px;
background:#f5f5f5;
}}

table {{
border-collapse: collapse;
background:white;
}}

td,th {{
padding:10px;
border:1px solid #ddd;
}}

.chart {{
background:white;
padding:20px;
margin-top:20px;
margin-bottom:40px;
}}

</style>

</head>

<body>

<h1>Virtualna Burza Dashboard</h1>

<p>Generated: {datetime.now()}</p>

<h2>Market Overview</h2>

<p><b>Top performing stock:</b> {top_stock['ticker']}</p>

<p><b>Portfolio return:</b> {portfolio_return:.2f}</p>

<p><b>Strategy win rate:</b> {win_rate:.0%}</p>


<h2>Summary</h2>

<table>

<tr>
<th>Ticker</th>
<th>BuyHold</th>
<th>Strategy</th>
<th>ADF p-value</th>
<th>Signal Today</th>
</tr>
"""

    for row in summary_rows:

        html += f"""
<tr>
<td>{row['ticker']}</td>
<td>{row['buyhold']:.2f}</td>
<td>{row['strategy']:.2f}</td>
<td>{row['adf']:.4f}</td>
<td>{row['signal']}</td>
</tr>
"""

    html += "</table>"

    for ticker,chart in charts.items():

        html += f"""

<h2>{ticker}</h2>

<div class="chart">
{chart['price']}
</div>

<div class="chart">
{chart['equity']}
</div>

"""

    html += "</body></html>"

    os.makedirs("output",exist_ok=True)

    with open("output/index.html","w") as f:
        f.write(html)


# =====================================================
# MAIN PIPELINE
# =====================================================

if __name__ == "__main__":

    date_from = "2020-01-01"

    date_to = (date.today() - timedelta(days=1)).isoformat()

    stocks = {

        "ERNT": {
            "isin":"HRERNTRA0000",
            "url":"https://zse.hr/hr/papir/310?isin=HRERNTRA0000"
        },

        "HT": {
            "isin":"HRHT00RA0005",
            "url":"https://zse.hr/hr/papir/310?isin=HRHT00RA0005"
        },

        "PODR": {
            "isin":"HRPODRRA0004",
            "url":"https://zse.hr/hr/papir/310?isin=HRPODRRA0004"
        }

    }

    results = {}

    summary_rows = []

    charts = {}

    with requests.Session() as session:

        for ticker,meta in stocks.items():

            html = fetch_html(session,meta["url"])

            tokens = extract_rest_tokens(html)

            for token in tokens:

                r = fetch_csv_with_token(
                    session,
                    token,
                    meta["isin"],
                    date_from,
                    date_to,
                    meta["url"]
                )

                if r.status_code == 200 and r.text.strip():

                    df_raw = pd.read_csv(StringIO(r.text),sep=";",decimal=",")

                    df_raw = clean_df(df_raw)

                    df = zse_api_to_internal_df(df_raw)

                    df = add_features(df)

                    df = run_strategy(df)

                    pval = arima_test(df)

                    signal = get_current_signal(df)

                    price_chart = generate_price_chart(df,ticker)

                    equity_chart = generate_equity_chart(df,ticker)

                    charts[ticker] = {
                        "price":price_chart,
                        "equity":equity_chart
                    }

                    summary_rows.append({

                        "ticker":ticker,
                        "buyhold":df["cum_buyhold"].iloc[-1],
                        "strategy":df["cum_strategy"].iloc[-1],
                        "adf":pval,
                        "signal":signal

                    })

                    break

            time.sleep(1)

    generate_html(summary_rows,charts)
