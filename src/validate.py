import pandas as pd

REQUIRED_COLUMNS = ["date", "open", "high", "low", "close"]


def validate_prices(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Prihvati ili 'volume' (dummy) ili 'turnover' (ZSE API)
    if "volume" not in df.columns and "turnover" not in df.columns:
        raise ValueError("Missing required column: 'volume' or 'turnover'")

    out = df.copy()

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if out["date"].isna().any():
        raise ValueError("Some 'date' values could not be parsed.")

    numeric_cols = ["open", "high", "low", "close"]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if out[numeric_cols].isna().any().any():
        raise ValueError("Some OHLC values could not be parsed as numeric.")

    out = (
        out.drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    if out.empty:
        raise ValueError("Validated dataframe is empty.")

    return out
