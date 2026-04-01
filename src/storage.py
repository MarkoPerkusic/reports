import duckdb
import pandas as pd


def get_connection(db_path: str = ":memory:") -> duckdb.DuckDBPyConnection:
    return duckdb.connect(database=db_path)


def save_prices(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> None:
    conn.register("prices_df", df)
    conn.execute("CREATE OR REPLACE TABLE prices AS SELECT * FROM prices_df")
