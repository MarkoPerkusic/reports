import pandas as pd


REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def validate_prices(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if out["date"].isna().any():
        raise ValueError("Some 'date' values could not be parsed.")

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if out[numeric_cols].isna().any().any():
        raise ValueError("Some OHLCV values could not be parsed as numeric.")

    out = (
        out.drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    if out.empty:
        raise ValueError("Validated dataframe is empty.")

    return out
