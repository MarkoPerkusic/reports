import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import CHARTS_DIR, SITE_DIR


def _ensure_site_dirs() -> None:
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def _save_charts(df: pd.DataFrame) -> None:
    # Equity curve
    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df["equity_curve"])
    plt.title("Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "equity_curve.png")
    plt.close()

    # Drawdown
    drawdown = df["equity_curve"] / df["equity_curve"].cummax() - 1
    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], drawdown)
    plt.title("Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "drawdown.png")
    plt.close()


def _write_metrics_files(metrics: dict) -> None:
    with open(SITE_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame([metrics]).to_csv(SITE_DIR / "metrics.csv", index=False)


def _write_html(metrics: dict) -> None:
    go_live = "PASS" if metrics["go_live"] else "FAIL"

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <title>Trading Metrics Report</title>
      <style>
        body {{
          font-family: Arial, sans-serif;
          max-width: 1000px;
          margin: 40px auto;
          padding: 0 20px;
        }}
        table {{
          border-collapse: collapse;
          width: 100%;
          margin-top: 20px;
        }}
        th, td {{
          border: 1px solid #ddd;
          padding: 10px;
          text-align: left;
        }}
        th {{
          background: #f4f4f4;
        }}
        .pass {{ color: green; font-weight: bold; }}
        .fail {{ color: red; font-weight: bold; }}
        img {{
          max-width: 100%;
          margin-top: 20px;
        }}
      </style>
    </head>
    <body>
      <h1>Trading Metrics Report</h1>
      <p>Deployment decision:
        <span class="{'pass' if metrics['go_live'] else 'fail'}">{go_live}</span>
      </p>

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

      <h2>Charts</h2>
      <img src="charts/equity_curve.png" alt="Equity Curve">
      <img src="charts/drawdown.png" alt="Drawdown">
    </body>
    </html>
    """

    with open(SITE_DIR / "index.html", "w", encoding="utf-8") as f:
        f.write(html)


def build_report(df: pd.DataFrame, metrics: dict) -> None:
    _ensure_site_dirs()
    _save_charts(df)
    _write_metrics_files(metrics)
    _write_html(metrics)
