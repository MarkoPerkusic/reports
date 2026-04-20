import duckdb
import pandas as pd


def get_connection(db_path: str = ":memory:") -> duckdb.DuckDBPyConnection:
    return duckdb.connect(database=db_path)


def save_prices(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame, ticker: str) -> None:
    out = df.copy()
    out.insert(0, "ticker", ticker)
    conn.register("prices_df", out)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prices AS SELECT * FROM prices_df WHERE 1=0;
        INSERT INTO prices SELECT * FROM prices_df;
    """)
