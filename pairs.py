import numpy as np
import pandas as pd


def top_pairs(corr: pd.DataFrame, n: int = 20) -> pd.Series:
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    pairs = corr.where(mask).stack().sort_values(ascending=False)
    return pairs.head(n)

def rank_pairs_by_spread_quality(prices: pd.DataFrame, candidate_pairs: pd.Series, top_n: int = 10) -> pd.DataFrame:
    rows = []

    for asset_1, asset_2 in candidate_pairs.index:
        pair_prices = prices[[asset_1, asset_2]].dropna()
        if len(pair_prices) < 100:
            continue

        spread = compute_spread(pair_prices, asset_1, asset_2)
        z = compute_zscore(spread, window=20)

        rows.append({
            "pair": f"{asset_1} - {asset_2}",
            "corr": candidate_pairs.loc[(asset_1, asset_2)],
            "spread_std": spread.std(),
            "zscore_std": z.std(skipna=True),
            "crossings": ((z.shift(1) * z) < 0).sum(),
        })

    df = pd.DataFrame(rows)
    return df.sort_values(["corr", "crossings"], ascending=[False, False]).head(top_n)

def compute_spread(prices: pd.DataFrame, asset_1: str, asset_2: str) -> pd.Series:
    log_prices = np.log(prices[[asset_1, asset_2]].dropna())

    x = log_prices[asset_2].values
    y = log_prices[asset_1].values

    beta = np.polyfit(x, y, 1)[0]

    spread = log_prices[asset_1] - beta * log_prices[asset_2]
    return spread


def compute_zscore(spread: pd.Series, window: int = 20) -> pd.Series:
    mean = spread.rolling(window).mean()
    std = spread.rolling(window).std()
    return (spread - mean) / std


def generate_signals(
    zscore: pd.Series,
    entry: float = 2.0,
    exit: float = 0.5,
    stop: float = 3.0
) -> pd.DataFrame:
    signals = pd.DataFrame(index=zscore.index)
    signals["zscore"] = zscore
    signals["long"] = zscore < -entry
    signals["short"] = zscore > entry
    signals["exit"] = zscore.abs() < exit
    signals["stop"] = zscore.abs() > stop
    return signals

def backtest_pairs_strategy(
    spread: pd.Series,
    signals: pd.DataFrame,
    max_holding: int = 15
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, dict[str, float]]:
    """
    Backtest a simple pairs trading strategy from spread and signal data.

    :param spread: Spread series for the selected pair.
    :type spread: pd.Series

    :param signals: DataFrame containing at least the columns
        ``long``, ``short``, ``exit``, and ``stop``.
    :type signals: pd.DataFrame

    :param max_holding: Maximum number of periods to hold a position.
    :type max_holding: int

    :returns: Positions, spread returns, strategy returns, cumulative pnl,
        and summary statistics.
    :rtype: tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, dict[str, float]]
    """
    required_cols = {"long", "short", "exit", "stop"}
    missing = required_cols - set(signals.columns)
    if missing:
        raise ValueError(f"Signals dataframe is missing required columns: {sorted(missing)}")

    positions = pd.DataFrame(index=signals.index)
    positions["position"] = 0.0

    current_position = 0.0
    holding_days = 0

    for t in signals.index:
        if current_position != 0.0:
            holding_days += 1
        else:
            holding_days = 0

        if current_position != 0.0 and (
            signals.loc[t, "exit"] or
            signals.loc[t, "stop"] or
            holding_days > max_holding
        ):
            current_position = 0.0
            holding_days = 0

        elif current_position == 0.0:
            if signals.loc[t, "long"]:
                current_position = 1.0
                holding_days = 0
            elif signals.loc[t, "short"]:
                current_position = -1.0
                holding_days = 0

        positions.loc[t, "position"] = current_position

    spread_returns = spread.diff().fillna(0.0)
    strategy_returns = positions["position"].shift(1).fillna(0.0) * spread_returns
    cum_pnl = strategy_returns.cumsum()

    total_return = strategy_returns.sum()
    mean_return = strategy_returns.mean()
    std_return = strategy_returns.std()
    sharpe = mean_return / std_return if std_return != 0 else np.nan

    stats = {
        "total_return": total_return,
        "mean_return": mean_return,
        "std_dev": std_return,
        "sharpe": sharpe,
    }

    return positions, spread_returns, strategy_returns, cum_pnl, stats