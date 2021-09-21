import pandas as pd
import numpy as np
import yfinance
from sklearn.preprocessing import MinMaxScaler
from tslearn.clustering import TimeSeriesKMeans, KShape
from tslearn.clustering import silhouette_score
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import to_time_series_dataset


class StockClusterer:
    """
    Time series clustering for stocks using KMeans or KShape algorithms.

    Parameters
    ----------
    stocks : list
        List of ticker symbols to cluster.
    n_clusters : int
        Number of clusters.
    change_method : str
        One of 'DailyReturn', 'MinMax', 'MinMaxCumulative', 'Cumulative', 'Z'.
    algo : str
        Clustering algorithm: 'KMeans' or 'KShape'.
    date_range : tuple
        (start_date, end_date) strings in 'YYYY-MM-DD' format.
    """

    SUPPORTED_METHODS = ("DailyReturn", "MinMax", "MinMaxCumulative", "Cumulative", "Z")
    SUPPORTED_ALGOS = ("KMeans", "KShape")

    def __init__(self, stocks, n_clusters, change_method, algo, date_range):
        if change_method not in self.SUPPORTED_METHODS:
            raise ValueError(f"change_method must be one of {self.SUPPORTED_METHODS}")
        if algo not in self.SUPPORTED_ALGOS:
            raise ValueError(f"algo must be one of {self.SUPPORTED_ALGOS}")

        self.stocks = stocks
        self.n_clusters = n_clusters
        self.change_method = change_method
        self.algo = algo
        self.date_range = date_range

        # populated after fit()
        self.clusters = None
        self.cluster_centers = None
        self.labels = None
        self.score = None
        self.unclustered = None
        self._df_final = None
        self._df_list = None

    def _download(self):
        df = yfinance.download(
            tickers=self.stocks,
            start=self.date_range[0],
            end=self.date_range[1],
            interval="1d",
            progress=False,
        )
        return df

    def _preprocess(self, df):
        method = self.change_method

        if method == "DailyReturn":
            df_change = df["Adj Close"].pct_change().copy()
            df_change = df_change[1:]
            df_change.dropna(how="all", axis=1, inplace=True)
            df_change.interpolate(axis=0, inplace=True)
            df_change.dropna(axis=0, inplace=True)
            return df_change.T

        elif method == "MinMax":
            df_change = df["Adj Close"].copy()
            df_change.dropna(how="all", axis=1, inplace=True)
            df_change.interpolate(axis=0, inplace=True)
            df_change.fillna(0, inplace=True)
            for col in df_change.columns:
                df_change[col] = MinMaxScaler().fit_transform(df_change[[col]])
            return df_change.T

        elif method == "MinMaxCumulative":
            df_change = df["Adj Close"].copy()
            df_change.dropna(how="all", axis=1, inplace=True)
            df_change.interpolate(axis=0, inplace=True)
            df_change.fillna(0, inplace=True)
            df_adj_close = df["Adj Close"][df_change.columns]
            df_adj_cum = (df_adj_close / df_adj_close.iloc[0]).copy()
            for col in df_change.columns:
                df_change[col] = MinMaxScaler().fit_transform(df_change[[col]])
            df_final = (df_change * df_adj_cum)
            df_final.dropna(how="all", axis=1, inplace=True)
            return df_final.T

        elif method == "Cumulative":
            df_change = (df["Adj Close"] / df["Adj Close"].iloc[0]).copy()
            df_change.dropna(how="all", axis=1, inplace=True)
            df_change.interpolate(axis=0, inplace=True)
            df_change.dropna(axis=0, inplace=True)
            return df_change.T

        elif method == "Z":
            df_change = df["Adj Close"].copy()
            df_change.dropna(how="all", axis=1, inplace=True)
            df_change.interpolate(axis=0, inplace=True)
            df_change.fillna(0, inplace=True)
            df_t = df_change.T
            mean_var = TimeSeriesScalerMeanVariance(mu=0.0, std=1.0).fit_transform(df_t)
            mean_var_l = [x.ravel() for x in mean_var]
            return pd.DataFrame(mean_var_l, columns=df_t.columns, index=df_t.index)

    def _cluster(self, df_final):
        ts_data = to_time_series_dataset(df_final)

        if self.algo == "KMeans":
            model = TimeSeriesKMeans(
                n_clusters=self.n_clusters, metric="dtw", random_state=0
            )
        else:
            model = KShape(n_clusters=self.n_clusters, random_state=0)

        model.fit(ts_data)
        labels = list(model.predict(ts_data))
        centers = model.cluster_centers_
        return labels, centers

    def fit(self):
        """Download data, preprocess, and fit the clustering model."""
        raw = self._download()
        self._df_final = self._preprocess(raw)

        labels, centers = self._cluster(self._df_final)

        # Map cluster id -> list of tickers
        clusters_df = pd.DataFrame({"cluster": labels}, index=self._df_final.index)
        clusters = {
            c: list(clusters_df[clusters_df["cluster"] == c].index)
            for c in set(labels)
        }

        center_list = [c.ravel() for c in centers]

        unclustered = []
        df_list = []

        for cluster_id, tickers in clusters.items():
            if len(tickers) > 1:
                df_t = self._df_final.T[tickers]
                df_t["cluster_center"] = center_list[cluster_id]
                df_list.append(df_t)
            else:
                unclustered.append(tickers[0])
                labels.remove(cluster_id)

        df_clustered = self._df_final.drop(unclustered)
        X_ts = to_time_series_dataset(df_clustered)

        try:
            score = silhouette_score(X_ts, labels, metric="dtw")
        except Exception:
            score = None

        self.clusters = clusters
        self.cluster_centers = center_list
        self.labels = labels
        self.score = score
        self.unclustered = unclustered
        self._df_list = df_list

        return self

    def get_results(self):
        """Return a summary dict of clustering results."""
        if self.clusters is None:
            raise RuntimeError("Call fit() before get_results().")
        return {
            "clusters": self.clusters,
            "labels": self.labels,
            "cluster_centers": self.cluster_centers,
            "silhouette_score": self.score,
            "unclustered": self.unclustered,
        }
