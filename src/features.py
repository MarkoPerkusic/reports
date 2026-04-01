import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["return"] = out["close"].pct_change()
    out["log_return"] = np.log(out["close"] / out["close"].shift(1))
    out["rolling_vol_20"] = out["return"].rolling(20).std() * np.sqrt(252)
    out["sma_20"] = out["close"].rolling(20).mean()
    out["sma_50"] = out["close"].rolling(50).mean()

    return out.dropna().reset_index(drop=True)
