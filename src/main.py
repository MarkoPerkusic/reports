from src.features import build_features
from src.ingest import load_data
from src.metrics import calculate_all_metrics
from src.report import build_report
from src.strategy import run_strategy
from src.validate import validate_prices
import sys


sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    all_data = load_data()

    if not all_data:
        print("Nema podataka — pipeline prekinut.")
        return

    all_results = {}

    for ticker, df_raw in all_data.items():
        print(f"\n[pipeline] Obrada: {ticker}")
        try:
            df = validate_prices(df_raw)
            features = build_features(df)
            strategy_df = run_strategy(features)
            metrics = calculate_all_metrics(strategy_df)
            all_results[ticker] = (strategy_df, metrics)
            print(f"[pipeline] {ticker} — go_live: {metrics['go_live']}")
        except Exception as e:
            print(f"[pipeline] {ticker} — greška: {e}")

    if all_results:
        build_report(all_results)
        print("\nPipeline completed successfully.")
    else:
        print("\nNijedan ticker nije uspješno obrađen.")


if __name__ == "__main__":
    main()
