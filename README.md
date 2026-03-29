# random-matrix-theory

# Purpose
Random Matrix Theory (RMT) provides a principled framework for distinguishing meaningful structure from randomness. Under the assumption of random returns, the eigenvalues of a correlation matrix follow the Marchenko–Pastur (MP) distribution. Eigenvalues within the MP bounds are consistent with noise, while those outside are more likely to reflect genuine market structure.

In financial markets, raw correlations are often unstable and heavily affected by estimation error. This is a major problem for strategies such as:

- portfolio optimization
- clustering and factor analysis
- statistical arbitrage
- pairs trading

RMT provides a principled way to distinguish signal from noise in the eigenvalue spectrum of the correlation matrix.

## Methodology

The workflow and results are in RMT.ipynb which consists of:

1. Download historical asset prices using `yfinance`
2. Compute log returns
3. Clean the returns matrix
4. Estimate the empirical correlation matrix
5. Compute the eigenvalue spectrum
6. Compare eigenvalues to the Marchenko–Pastur bounds
7. Identify signal eigenvalues outside the MP bulk
8. Reconstruct a denoised correlation matrix
9. Select candidate pairs from the denoised matrix
10. Build a simple pairs trading strategy using spread z-scores


## Future Improvements
This project uses a relatively simple backtesting setup and does not yet include:
- transaction costs
- slippage
- rolling hedge ratio estimation
- portfolio level optimization
- regime filtering

Possible extensions include:
- rolling-window RMT analysis
- volatility filters
- multi-pair portfolio construction
- full backtesting with transaction costs and performance metrics
