import pandas as pd
import numpy as np


def load_data() -> pd.DataFrame:
    np.random.seed(42)

    dates = pd.date_range(start="2023-01-01", periods=200, freq="D")

    price = 100 + np.cumsum(np.random.normal(0, 1, len(dates)))

    df = pd.DataFrame({
        "date": dates,
        "close": price
    })

    # generiraj OHLC oko close
    df["open"] = df["close"] + np.random.normal(0, 0.5, len(df))
    df["high"] = df[["open", "close"]].max(axis=1) + np.abs(np.random.normal(0, 1, len(df)))
    df["low"] = df[["open", "close"]].min(axis=1) - np.abs(np.random.normal(0, 1, len(df)))
    df["volume"] = np.random.randint(1000, 10000, len(df))

    # reorder columns
    df = df[["date", "open", "high", "low", "close", "volume"]]

    return df
