# Time Series Stock Clustering

Unsupervised time series clustering of stocks to identify groups of equities with similar price behavior over a given period.

## Overview

Given a list of stock tickers and a date range, this project downloads historical price data via `yfinance`, applies a normalization method to the price series, and runs a time series clustering algorithm to group stocks by behavioral similarity.

The core idea is that stocks within the same cluster move in similar ways - even if their absolute prices differ. This can be useful for portfolio diversification (avoid holding multiple stocks in the same cluster), sector analysis, or identifying regime shifts across asset classes.

## Methodology

### Normalization Methods

Raw prices are not directly comparable across stocks. Before clustering, each stock's series is transformed using one of several methods:

| Method | Description |
|---|---|
| `DailyReturn` | Daily % change in adjusted close — focuses on short-term co-movement |
| `Cumulative` | Return relative to start price — captures overall trend shape |
| `MinMax` | Scales each series to [0, 1] — normalizes by historical range |
| `MinMaxCumulative` | Product of MinMax and cumulative return — blend of magnitude and trend |
| `Z` | Z-score via mean-variance scaling — required for KShape |

### Clustering Algorithms

**KMeans (DTW)** — Uses Dynamic Time Warping as the distance metric instead of Euclidean distance. DTW allows for temporal flexibility when comparing two series, meaning similar patterns that are slightly shifted in time are still recognized as similar. Generally works with any normalization method.

**KShape** — A shape-based clustering algorithm that uses cross-correlation to measure similarity. It internally z-normalizes all input regardless of preprocessing, so `change_method="Z"` should be used. KShape is particularly effective at capturing the overall shape of a price trajectory.

### Evaluation

Cluster quality is scored using the **Silhouette Score**, which measures how similar each stock is to its own cluster compared to other clusters. Scores range from -1 to 1, with higher values indicating more distinct, well-separated clusters.

## Usage

```python
from clustering import StockClusterer

clusterer = StockClusterer(
    stocks=["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
    n_clusters=3,
    change_method="Cumulative",
    algo="KMeans",
    date_range=("2023-01-01", "2024-01-01"),
)

clusterer.fit()
results = clusterer.get_results()
print(results["clusters"])
clusterer.plot()
```

## Example

See `example.ipynb` for a walkthrough using 15 stocks across tech, financials, and energy. The notebook covers setup, fitting, inspecting cluster assignments and silhouette score, and visualizing the results.

## Dependencies

See `requirements.txt`.
