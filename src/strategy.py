import pandas as pd


def run_strategy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["signal"] = 1
    out["position"] = out["signal"].shift(1).fillna(0)
    out["strategy_return"] = out["position"] * out["return"]
    out["equity_curve"] = (1 + out["strategy_return"]).cumprod()

    return out
