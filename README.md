# ts-stock-clustering

Time series clustering of stocks using KMeans and KShape algorithms.

## Overview

Groups a set of stock tickers into clusters based on the similarity of their price behavior over a given time period. Supports multiple normalization methods and uses dynamic time warping (DTW) as the distance metric.

## Usage

```python
from clustering import StockClusterer

clusterer = StockClusterer(
    stocks=["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
    n_clusters=3,
    change_method="DailyReturn",  # or MinMax, Cumulative, Z
    algo="KMeans",                # or KShape
    date_range=("2023-01-01", "2024-01-01"),
)

clusterer.fit()
results = clusterer.get_results()
print(results["clusters"])
```

## Change Methods

| Method | Description |
|---|---|
| `DailyReturn` | Daily percentage change in adjusted close |
| `MinMax` | Min-max normalized adjusted close |
| `Cumulative` | Cumulative return from start date |
| `MinMaxCumulative` | Product of min-max and cumulative return |
| `Z` | Z-score normalized via mean-variance scaling |

## Dependencies

See `requirements.txt`.
