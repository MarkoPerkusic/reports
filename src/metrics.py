import numpy as np
import pandas as pd

from src.config import RISK_FREE_ANNUAL, THRESHOLDS, TRADING_DAYS


def _risk_free_daily() -> float:
    return (1 + RISK_FREE_ANNUAL) ** (1 / TRADING_DAYS) - 1


def calculate_sharpe(returns: pd.Series) -> float:
    excess = returns - _risk_free_daily()
    std = excess.std()
    if std == 0 or pd.isna(std):
        return np.nan
    return float(excess.mean() / std * np.sqrt(TRADING_DAYS))


def calculate_sortino(returns: pd.Series) -> float:
    excess = returns - _risk_free_daily()
    downside = excess.copy()
    downside[downside > 0] = 0
    downside_dev = np.sqrt((downside**2).mean())
    if downside_dev == 0 or pd.isna(downside_dev):
        return np.nan
    return float(excess.mean() / downside_dev * np.sqrt(TRADING_DAYS))


def calculate_var(returns: pd.Series, percentile: int = 5) -> float:
    return float(np.percentile(returns.dropna(), percentile))


def calculate_cvar(returns: pd.Series, var_95: float) -> float:
    tail = returns[returns <= var_95]
    if tail.empty:
        return np.nan
    return float(tail.mean())


def calculate_drawdown(returns: pd.Series) -> pd.Series:
    equity = (1 + returns).cumprod()
    rolling_max = equity.cummax()
    return equity / rolling_max - 1


def calculate_max_drawdown(returns: pd.Series) -> float:
    dd = calculate_drawdown(returns)
    return float(dd.min())


def calculate_calmar(returns: pd.Series) -> float:
    cumulative = (1 + returns).cumprod()
    if cumulative.empty:
        return np.nan
    cagr = cumulative.iloc[-1] ** (TRADING_DAYS / len(returns)) - 1
    mdd = abs(calculate_max_drawdown(returns))
    if mdd == 0 or pd.isna(mdd):
        return np.nan
    return float(cagr / mdd)


def calculate_expectancy(returns: pd.Series) -> tuple[float, float]:
    trade_returns = returns.dropna()
    if trade_returns.empty:
        return np.nan, np.nan

    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]

    win_rate = len(wins) / len(trade_returns)
    avg_win = wins.mean() if not wins.empty else 0.0
    avg_loss = abs(losses.mean()) if not losses.empty else 0.0

    expectancy = float(trade_returns.mean())
    expectancy_r = float((win_rate * avg_win) - ((1 - win_rate) * avg_loss))
    return expectancy, expectancy_r


def calculate_mae(df: pd.DataFrame) -> float:
    """
    Simplified MAE proxy for daily-bar data.

    True MAE is trade-path based. Since this pipeline starts with daily bars and
    a simple strategy, we estimate adverse excursion from intraday low relative
    to previous close for long exposure days.
    """
    working = df.copy()
    long_days = working[working["position"] > 0].copy()
    if long_days.empty:
        return np.nan

    long_days["prev_close"] = long_days["close"].shift(1)
    long_days = long_days.dropna(subset=["prev_close"])
    if long_days.empty:
        return np.nan

    adverse = (long_days["low"] - long_days["prev_close"]) / long_days["prev_close"]
    adverse = adverse[adverse < 0].abs()

    if adverse.empty:
        return 0.0

    return float(adverse.mean())


def deployment_decision(metrics: dict) -> dict:
    checks = {
        "sharpe_pass": metrics["sharpe"] >= THRESHOLDS["sharpe_min"],
        "sortino_pass": metrics["sortino"] >= THRESHOLDS["sortino_min"],
        "var_pass": abs(metrics["var_95"]) <= THRESHOLDS["var_95_max_abs"],
        "cvar_pass": (
            False if metrics["var_95"] == 0 else abs(metrics["cvar_95"]) / abs(metrics["var_95"]) <= THRESHOLDS["cvar_var_ratio_max"]
        ),
        "expectancy_pass": metrics["expectancy_r"] >= THRESHOLDS["expectancy_r_min"],
        "max_drawdown_pass": abs(metrics["max_drawdown"]) <= THRESHOLDS["max_drawdown_max_abs"],
        "calmar_pass": metrics["calmar"] >= THRESHOLDS["calmar_min"],
    }
    checks["go_live"] = all(checks.values())
    return checks


def calculate_all_metrics(df: pd.DataFrame) -> dict:
    returns = df["strategy_return"].dropna()

    sharpe = calculate_sharpe(returns)
    sortino = calculate_sortino(returns)
    var_95 = calculate_var(returns, 5)
    cvar_95 = calculate_cvar(returns, var_95)
    expectancy, expectancy_r = calculate_expectancy(returns)
    max_drawdown = calculate_max_drawdown(returns)
    mae = calculate_mae(df)
    calmar = calculate_calmar(returns)

    metrics = {
        "sharpe": sharpe,
        "sortino": sortino,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "expectancy": expectancy,
        "expectancy_r": expectancy_r,
        "max_drawdown": max_drawdown,
        "mae": mae,
        "calmar": calmar,
    }

    metrics.update(deployment_decision(metrics))
    return metrics
