import numpy as np

def mp_bounds(n_assets: int, n_obs: int) -> tuple[float, float]:
    """
    Compute the Marchenko-Pastur support for a sample correlation matrix.

    :param n_assets: Number of assets, i.e. the dimension of the correlation matrix.
    :type n_assets: int

    :param n_obs: Number of observations used to estimate the matrix.
    :type n_obs: int

    :returns: Lower and upper Marchenko-Pastur bounds.
    :rtype: tuple[float, float]
    :raises ValueError: If ``n_assets`` or ``n_obs`` is non-positive.
    """

    if n_assets <= 0 or n_obs <= 0:
        raise ValueError("n_assets and n_obs must be positive")

    q = n_assets / n_obs

    # The variance of MP-distribution for a correlation matrix is just 1 since
    # the variance of the underlying process is unknown
    lower_bound = (1 - np.sqrt(q)) ** 2
    upper_bound = (1 + np.sqrt(q)) ** 2

    return [lower_bound, upper_bound]