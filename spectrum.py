from __future__ import annotations

import numpy as np
import pandas as pd


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the sample correlation matrix from returns.

    :param returns: DataFrame of asset returns indexed by date.
    :type returns: pd.DataFrame

    :returns: Correlation matrix.
    :rtype: pd.DataFrame
    :raises ValueError: If the input dataframe is empty.
    """
    if returns.empty:
        raise ValueError("Returns dataframe is empty")

    return returns.corr()


def eigen_spectrum(matrix: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the eigenvalue spectrum of a symmetric matrix.

    Eigenvalues and eigenvectors are returned in descending order
    of eigenvalues.

    :param matrix: Symmetric matrix, typically a correlation matrix.
    :type matrix: pd.DataFrame | np.ndarray

    :returns: Tuple of eigenvalues and eigenvectors.
    :rtype: tuple[np.ndarray, np.ndarray]
    :raises ValueError: If the input is not a square 2D matrix.
    """
    arr = matrix.to_numpy() if isinstance(matrix, pd.DataFrame) else np.asarray(matrix)

    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Input matrix must be square")

    eigenvalues, eigenvectors = np.linalg.eigh(arr)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    return eigenvalues, eigenvectors