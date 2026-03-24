import yfinance as yf
import numpy as np
import pandas as pd

def fetch_prices(tickers: list[str], start: str, end: str, auto_adjust: bool = True):
    """
    Download closing price data for a list of tickers.

    :param tickers: List of asset tickers.
    :type tickers: list[str]

    :param start: Start date in ``YYYY-MM-DD`` format.
    :type start: str

    :param end: End date in ``YYYY-MM-DD`` format.
    :type end: str

    :param auto_adjust: Whether to use adjusted prices.
    :type auto_adjust: bool

    :returns: DataFrame of prices indexed by date, with tickers as columns.
    :rtype: pd.DataFrame
    :raises ValueError: If ``tickers`` is empty or no price data is downloaded.
    """
    data = yf.download(
        tickers = tickers,
        start = start,
        end = end,
        auto_adjust = auto_adjust,
        progress = False,
    )

    if data.empty:
        raise ValueError("No price data was downloaded")
    
    # yfinance returns a multi-index columns dataframe for multiple tickers
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            prices = data["Close"].copy()
        else:
            # with auto_adjust=True, "Close" is still usually present, but guard anyway
            prices = data.xs(data.columns.levels[0][0], axis=1, level=0).copy()
    else:
        prices = data.to_frame(name=tickers[0]) if isinstance(data, pd.Series) else data.copy()
        if "Close" in prices.columns:
            prices = prices[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.sort_index()
    prices = prices.dropna(axis = 1, how = "all")

    if prices.empty:
        raise ValueError("Price dataframe is empty after cleaning")

    return prices

def compute_returns(prices: pd.DataFrame, log: bool = True) -> pd.DataFrame:
    """
    Compute returns from price data.

    :param prices: DataFrame of asset prices indexed by date.
    :type prices: pd.DataFrame

    :param log: If ``True``, compute log returns. Otherwise compute simple returns.
    :type log: bool

    :returns: DataFrame of asset returns indexed by date.
    :rtype: pd.DataFrame
    :raises ValueError: If the input price dataframe is empty.
    """
    if log:
        returns = np.log(prices / prices.shift(1))
    else:
        returns = prices.pct_change()

    return returns.dropna(how = "all")

def clean_returns(returns: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """
    Clean a returns matrix for correlation and RMT analysis.

    Columns with too many missing values are removed first. Remaining rows
    containing missing values are then dropped to produce a complete matrix.

    :param returns: DataFrame of asset returns indexed by date.
    :type returns: pd.DataFrame

    :param threshold: Minimum fraction of non missing observations
        required to keep an asset column.
    :type threshold: float
    
    :returns: Cleaned returns matrix with no missing values.
    :rtype: pd.DataFrame
    :raises ValueError: If the input dataframe is empty or cleaning removes all data.
    """
    if returns.empty:
        raise ValueError("Returns is empty")
    
    cleaned = returns.dropna(axis = 1, thresh=threshold)
    cleaned = cleaned.dropna(axis = 0, how = "any")

    if cleaned.empty:
        raise ValueError("No data left after cleaning returns")
    
    return cleaned
    





