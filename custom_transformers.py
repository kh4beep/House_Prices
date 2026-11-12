import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer


def on_columns(transformer, columns: list[str]) -> ColumnTransformer:
    """
    Apply a transformer to the given columns and pass the rest through unchanged.
    Note: ColumnTransformer puts the transformed columns first, so column order changes.
    Output names stay unprefixed (verbose_feature_names_out=False), so they must be unique.
    """
    return ColumnTransformer(
        [('t', transformer, columns)],
        remainder='passthrough',
        verbose_feature_names_out=False,
    )


def _mode(s: pd.Series):
    """Most frequent value; ties resolve to the first in sorted order, so it is deterministic."""
    m = s.mode()
    return m.iloc[0] if len(m) else np.nan


class GroupStatImputer(BaseEstimator, TransformerMixin):
    """
    Fill NaN in target_col with a per-group statistic learned at fit.
    Groups with fewer than min_count_group rows fall back to the global statistic.
    """
    def __init__(self, *, target_col: str, group_cols: list[str], method: str, min_count_group: int = 10) -> None:
        self.target_col = target_col
        self.group_cols = group_cols
        self.method = method
        self.min_count_group = min_count_group


    def _stat(self, s: pd.Series):
        return _mode(s) if self.method == 'mode' else s.agg(self.method)


    def fit(self, X, y=None):
        group_cols = list(self.group_cols)
        stats = X.groupby(group_cols)[self.target_col].agg(
            stat=(lambda s: _mode(s)) if self.method == 'mode' else self.method,
            count='count',
        )
        self.group_stat_ = stats.loc[stats['count'] >= self.min_count_group, 'stat']
        self.global_stat_ = self._stat(X[self.target_col])
        self.feature_names_in_ = np.asarray(X.columns)
        return self


    def transform(self, X):
        check_is_fitted(self, 'group_stat_')
        X = X.copy()
        group_cols = list(self.group_cols)
        fill = X[group_cols].merge(
            self.group_stat_, left_on=group_cols, right_index=True, how='left'
        )['stat']
        X[self.target_col] = X[self.target_col].fillna(fill).fillna(self.global_stat_)
        return X


    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return self.feature_names_in_



class EstimatorImputer(BaseEstimator, TransformerMixin):
    """
    Fill NaN in target_col with predictions of an estimator trained on rows where it is known.
    The estimator sees every other column that reaches this step, so its feature set is
    defined by the pipeline position; wrap it in a Pipeline to encode or select inputs.
    """
    def __init__(self, *, target_col: str, estimator: BaseEstimator) -> None:
        self.target_col = target_col
        self.estimator = estimator


    def fit(self, X: pd.DataFrame, y=None) -> 'EstimatorImputer':
        X = X[X[self.target_col].notna()]
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X.drop(columns=[self.target_col]), X[self.target_col])
        self.feature_names_in_ = np.asarray(X.columns)
        return self


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, 'estimator_')
        X = X.copy()
        missing = X[self.target_col].isna()
        if missing.any():
            fill = self.estimator_.predict(X[missing].drop(columns=[self.target_col]))
            X.loc[missing, self.target_col] = fill
        return X


    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return self.feature_names_in_



class SafeCategoryTransformer(BaseEstimator, TransformerMixin):
    """
    Learns valid categories during fit. During transform, maps any value 
    not seen in fit (and not NaN) to NaN, then casts to pandas 'category'.
    """
    def __init__(self) -> None:
        pass

    def fit(self, X: pd.DataFrame, y=None) -> 'SafeCategoryTransformer':
        X = pd.DataFrame(X)
        self.categories_ = {}
        for col in X.columns:
            self.categories_[col] = list(X[col].dropna().unique())
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, 'categories_')
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            if col in self.categories_:
                allowed_set = set(self.categories_[col])
                X[col] = pd.Categorical(
                    [val if (pd.isna(val) or val in allowed_set) else np.nan for val in X[col]],
                    categories=self.categories_[col],
                    ordered=False
                )
        return X

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return self.feature_names_in_



class MyTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, *, group_col: str, name: str, smoothing: float = 0.1) -> None:
        self.group_col = group_col
        self.name = name
        self.smoothing = smoothing


    @staticmethod
    def _stats(X: pd.DataFrame, y: np.ndarray, group_col: str) -> pd.DataFrame:
        return X[[group_col]].assign(__y__=y).groupby(group_col)['__y__'].agg(['count', 'sum'])


    def _encode(self, X: pd.DataFrame, stats: pd.DataFrame, prior: float) -> np.ndarray:
        joined = X[[self.group_col]].join(stats, on=self.group_col)
        n = joined['count'].fillna(0).to_numpy()
        s = joined['sum'].fillna(0).to_numpy()
        return (s + self.smoothing * prior) / (n + self.smoothing)


    def fit(self, X: pd.DataFrame, y=None) -> 'MyTargetEncoder':
        if y is None:
            raise ValueError('MyTargetEncoder requires y')
        y = np.asarray(y, dtype=float)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.stats_ = self._stats(X, y, self.group_col)
        self.prior_ = float(y.mean())
        return self


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, 'stats_')
        values = self._encode(X, self.stats_, self.prior_)
        return X.assign(**{self.name: values})


    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        if y is None:
            raise ValueError('MyTargetEncoder requires y')
        y_arr = np.asarray(y, dtype=float)
        stats = self._stats(X, y_arr, self.group_col)
        joined = X[[self.group_col]].join(stats, on=self.group_col)
        n = joined['count'].to_numpy(dtype=float) - 1
        s = joined['sum'].to_numpy(dtype=float) - y_arr
        prior = float(y_arr.mean())
        values = (s + self.smoothing * prior) / (n + self.smoothing)
        self.fit(X, y_arr)
        return X.assign(**{self.name: values})


    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.concatenate([self.feature_names_in_, [self.name]])



class CombinedFeatureCreator(BaseEstimator, TransformerMixin):
    def __init__(self, features, feature_name, operation='sum'):
        self.features = features
        self.feature_name = feature_name
        self.operation = operation


    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self


    def transform(self, X):
        X = X.copy()
        if self.operation == 'sum':
            X[self.feature_name] = X[self.features].sum(axis=1)
        elif self.operation == 'div':
            X[self.feature_name] = X[self.features[0]] / X[self.features[1]].replace(0, np.nan)
        elif self.operation == 'product':
            X[self.feature_name] = X[self.features].prod(axis=1)
        elif self.operation == 'diff':
            X[self.feature_name] = X[self.features[0]] - X[self.features[1]]
        return X


    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.concatenate([self.feature_names_in_, [self.feature_name]])


class GroupFeatureCreator(BaseEstimator, TransformerMixin):
    def __init__(self, features, feature_name, operation='sum'):
        self.features = features
        self.feature_name = feature_name
        self.operation = operation


    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self


    def transform(self, X):
        X = X.copy()
        if self.operation == 'sum':
            X[self.feature_name] = X[self.features].sum(axis=1)
        elif self.operation == 'div':
            X[self.feature_name] = X[self.features[0]] / X[self.features[1]].replace(0, np.nan)
        elif self.operation == 'product':
            X[self.feature_name] = X[self.features].prod(axis=1)
        elif self.operation == 'diff':
            X[self.feature_name] = X[self.features[0]] - X[self.features[1]]
        return X


    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.concatenate([self.feature_names_in_, [self.feature_name]])



class GroupByFeatureCreator(BaseEstimator, TransformerMixin):
    def __init__(self, *, group_cols, source_col, feature_name, method='mean'):
        self.group_cols = group_cols
        self.source_col = source_col
        self.feature_name = feature_name
        self.method = method


    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.group_stats_ = X.groupby(self.group_cols)[self.source_col].agg(self.method)
        self.global_stat_ = X[self.source_col].agg(self.method)
        return self


    def transform(self, X):
        check_is_fitted(self, 'group_stats_')
        X = X.copy()
        values = X[self.group_cols].merge(
            self.group_stats_.rename('__group_stat__'),
            left_on=self.group_cols,
            right_index=True,
            how='left',
            sort=False
        )['__group_stat__']
        X[self.feature_name] = values.fillna(self.global_stat_).to_numpy()
        return X


    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.concatenate([self.feature_names_in_, [self.feature_name]])



class MaxValueClipper(BaseEstimator, TransformerMixin):
    def __init__(self, features, percentile):
        self.percentile = percentile
        self.features = features


    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.thresholds_ = X[self.features].quantile(self.percentile / 100).to_dict()
        return self


    def transform(self, X):
        check_is_fitted(self, 'thresholds_')
        X = X.copy()
        for col, threshold in self.thresholds_.items():
            X[col] = X[col].clip(upper=threshold)
        return X


    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return self.feature_names_in_
