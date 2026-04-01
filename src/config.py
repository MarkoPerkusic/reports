from pathlib import Path

SITE_DIR = Path("site")
CHARTS_DIR = SITE_DIR / "charts"

DB_PATH = ":memory:"

SYMBOL = "AAPL"
LOOKBACK_YEARS = 3
INTERVAL = "1d"

RISK_FREE_ANNUAL = 0.04
TRADING_DAYS = 252

THRESHOLDS = {
    "sharpe_min": 1.2,
    "sortino_min": 2.0,
    "var_95_max_abs": 0.02,
    "cvar_var_ratio_max": 2.0,
    "expectancy_r_min": 0.1,
    "max_drawdown_max_abs": 0.25,
    "calmar_min": 1.5,
}
