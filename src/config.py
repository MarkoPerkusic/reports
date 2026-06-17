from pathlib import Path
import os

SITE_DIR = Path("site")
CHARTS_DIR = SITE_DIR / "charts"

USE_DUMMY_DATA = False

PAW_BASE_URL = os.getenv(
    "PAW_BASE_URL",
    "https://MarPer.pythonanywhere.com"
)

STOCKS = {
    "ERNT": {
        "isin": "HRERNTRA0000",
        "page_url": "https://zse.hr/hr/papir/310?isin=HRERNTRA0000",
    },
    "HT": {
        "isin": "HRHT00RA0005",
        "page_url": "https://zse.hr/hr/papir/310?isin=HRHT00RA0005",
    },
    "PODR": {
        "isin": "HRPODRRA0004",
        "page_url": "https://zse.hr/hr/papir/310?isin=HRPODRRA0004",
    },
}

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