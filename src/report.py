import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import CHARTS_DIR, SITE_DIR


def _ensure_site_dirs() -> None:
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def _save_charts(ticker: str, df: pd.DataFrame) -> None:
    # Equity curve
    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df["equity_curve"])
    plt.title(f"Equity Curve — {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / f"{ticker}_equity_curve.png")
    plt.close()

    # Drawdown
    drawdown = df["equity_curve"] / df["equity_curve"].cummax() - 1
    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], drawdown)
    plt.title(f"Drawdown — {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / f"{ticker}_drawdown.png")
    plt.close()


def _ticker_html_block(ticker: str, metrics: dict) -> str:
    go_live = "PASS" if metrics["go_live"] else "FAIL"
    css_class = "pass" if metrics["go_live"] else "fail"

    return f"""
    <section>
      <h2>{ticker}</h2>
      <p>Deployment decision: <span class="{css_class}">{go_live}</span></p>
      <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Sharpe</td><td>{metrics['sharpe']:.4f}</td></tr>
        <tr><td>Sortino</td><td>{metrics['sortino']:.4f}</td></tr>
        <tr><td>VaR 95%</td><td>{metrics['var_95']:.4%}</td></tr>
        <tr><td>CVaR 95%</td><td>{metrics['cvar_95']:.4%}</td></tr>
        <tr><td>Expectancy</td><td>{metrics['expectancy']:.4%}</td></tr>
        <tr><td>Expectancy (R)</td><td>{metrics['expectancy_r']:.4f}R</td></tr>
        <tr><td>Max Drawdown</td><td>{metrics['max_drawdown']:.4%}</td></tr>
        <tr><td>MAE</td><td>{metrics['mae']:.4%}</td></tr>
        <tr><td>Calmar</td><td>{metrics['calmar']:.4f}</td></tr>
      </table>
      <img src="charts/{ticker}_equity_curve.png" alt="Equity Curve {ticker}">
      <img src="charts/{ticker}_drawdown.png" alt="Drawdown {ticker}">
    </section>
    <hr>
    """


def _write_html(all_metrics: dict[str, dict]) -> None:
    body = ""
    for ticker, metrics in all_metrics.items():
        body += _ticker_html_block(ticker, metrics)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Trading Metrics Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 1000px; margin: 40px auto; padding: 0 20px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
    th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
    th {{ background: #f4f4f4; }}
    .pass {{ color: green; font-weight: bold; }}
    .fail {{ color: red; font-weight: bold; }}
    img {{ max-width: 100%; margin-top: 20px; }}
    hr {{ margin: 40px 0; }}
  </style>
</head>
<body>
  <h1>Trading Metrics Report</h1>
  {body}
</body>
</html>"""

    with open(SITE_DIR / "index.html", "w", encoding="utf-8") as f:
        f.write(html)


def _write_metrics_files(all_metrics: dict[str, dict]) -> None:
    with open(SITE_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    rows = [{"ticker": t, **m} for t, m in all_metrics.items()]
    pd.DataFrame(rows).to_csv(SITE_DIR / "metrics.csv", index=False)


def build_report(all_results: dict[str, tuple[pd.DataFrame, dict]]) -> None:
    """
    all_results = {
        "ERNT": (strategy_df, metrics),
        "HT":   (strategy_df, metrics),
        ...
    }
    """
    _ensure_site_dirs()

    all_metrics = {}
    for ticker, (strategy_df, metrics) in all_results.items():
        _save_charts(ticker, strategy_df)
        all_metrics[ticker] = metrics

    _write_metrics_files(all_metrics)
    _write_html(all_metrics)
