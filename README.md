# random-matrix-theory

# Purpose
Financial correlation matrices are often noisy, especially when the number of assets is comparable to the number of observations. This noise leads to unstable relationships and spurious signals, which can degrade trading performance.

Random Matrix Theory (RMT) provides a principled framework for distinguishing meaningful structure from randomness. Under the assumption of random returns, the eigenvalues of a correlation matrix follow the Marchenko–Pastur (MP) distribution. Eigenvalues within the MP bounds are consistent with noise, while those outside are more likely to reflect genuine market structure.

This project applies RMT to denoise correlation matrices and improve the signal-to-noise ratio in financial data. The cleaned structure is then used to construct more reliable statistical arbitrage strategies, such as pairs trading, with the goal of extracting robust alpha from noisy markets.