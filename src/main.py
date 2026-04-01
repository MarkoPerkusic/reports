from src.config import DB_PATH
from src.features import build_features
from src.ingest import load_data
from src.metrics import calculate_all_metrics
from src.report import build_report
from src.storage import get_connection, save_prices
from src.strategy import run_strategy
from src.validate import validate_prices


def main() -> None:
    df = load_data()
    df = validate_prices(df)

    conn = get_connection(DB_PATH)
    save_prices(conn, df)

    features = build_features(df)
    strategy_df = run_strategy(features)
    metrics = calculate_all_metrics(strategy_df)

    build_report(strategy_df, metrics)

    print("Pipeline completed successfully.")
    print(metrics)


if __name__ == "__main__":
    main()
