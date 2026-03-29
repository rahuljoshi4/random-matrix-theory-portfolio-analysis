import numpy as np
import pandas as pd


def denoise_correlation_matrix(
    corr: pd.DataFrame,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    lambda_max: float
) -> pd.DataFrame:
    """
    Reconstruct a denoised correlation matrix by keeping signal eigenvalues
    and replacing noise eigenvalues with their average.

    :param corr: Original correlation matrix.
    :type corr: pd.DataFrame

    :param eigenvalues: Eigenvalues of the correlation matrix.
    :type eigenvalues: np.ndarray

    :param eigenvectors: Eigenvectors of the correlation matrix.
    :type eigenvectors: np.ndarray

    :param lambda_max: Upper Marchenko-Pastur bound.
    :type lambda_max: float

    :returns: Denoised correlation matrix.
    :rtype: pd.DataFrame
    """
    signal_mask = eigenvalues > lambda_max
    noise_mask = ~signal_mask

    cleaned_eigenvalues = eigenvalues.copy()

    if np.any(noise_mask):
        cleaned_eigenvalues[noise_mask] = np.mean(eigenvalues[noise_mask])

    cleaned = eigenvectors @ np.diag(cleaned_eigenvalues) @ eigenvectors.T
    cleaned = (cleaned + cleaned.T) / 2

    # rescale back to a correlation matrix
    diag = np.sqrt(np.diag(cleaned))
    cleaned = cleaned / np.outer(diag, diag)
    np.fill_diagonal(cleaned, 1.0)

    return pd.DataFrame(cleaned, index=corr.index, columns=corr.columns)