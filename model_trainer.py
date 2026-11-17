import pickle

import numpy as np
import pandas as pd

from typing import Any
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


class ModelTrainer(RegressorMixin, BaseEstimator):
    """Wrap an estimator with target clipping, log1p and optional standardization."""
    def __init__(
        self,
        *,
        estimator: Any,
        is_scale_target: bool = False,
        winsorization_value: float | None = None,
        fit_kwargs: dict | None = None,
    ):
        self.estimator = estimator
        self.is_scale_target = is_scale_target
        self.winsorization_value = winsorization_value
        self.fit_kwargs = fit_kwargs


    @staticmethod
    def _validate_target(y) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if y.ndim != 1 or not len(y) or not np.isfinite(y).all() or (y < 0).any():
            raise ValueError('y must be a non-empty one-dimensional array of finite non-negative values')
        return y


    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray, **fit_params) -> 'ModelTrainer':
        """Fit on transformed targets; eval_set targets are transformed without clipping."""
        y = self._validate_target(y)
        if len(X) != len(y):
            raise ValueError('X and y must have the same number of rows')
        if self.winsorization_value is not None:
            y = np.minimum(y, self.winsorization_value)
        y = np.log1p(y)
        target_scaler = StandardScaler() if self.is_scale_target else None
        if target_scaler is not None:
            y = target_scaler.fit_transform(y.reshape(-1, 1)).ravel()

        model = clone(self.estimator, safe=False)
        fit_kwargs = dict(self.fit_kwargs or {})
        fit_kwargs.update(fit_params)

        if 'eval_set' in fit_kwargs:
            eval_set = fit_kwargs['eval_set']
            if isinstance(eval_set, tuple):
                eval_set = [eval_set]
            transformed_eval_set = []
            for X_val, y_val in eval_set:
                y_val = self._validate_target(y_val)
                if len(X_val) != len(y_val):
                    raise ValueError('Validation X and y must have the same number of rows')
                y_val = np.log1p(y_val)
                if target_scaler is not None:
                    y_val = target_scaler.transform(y_val.reshape(-1, 1)).ravel()
                transformed_eval_set.append((X_val, y_val))
            fit_kwargs['eval_set'] = transformed_eval_set

        model.fit(X, y, **fit_kwargs)
        artifacts = model.get_training_artifacts() if hasattr(model, 'get_training_artifacts') else {}
        self.model_ = model
        self.target_scaler_ = target_scaler
        self.artifacts_ = artifacts
        self.n_features_in_ = X.shape[1]
        if hasattr(self, 'feature_names_in_'):
            del self.feature_names_in_
        if isinstance(X, pd.DataFrame) and all(isinstance(col, str) for col in X.columns):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self


    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return non-negative prices after reversing target scaling and log1p."""
        check_is_fitted(self, 'model_')
        if X.shape[1] != self.n_features_in_:
            raise ValueError('X must have the same number of features as at fit')
        if hasattr(self, 'feature_names_in_') and isinstance(X, pd.DataFrame):
            if not np.array_equal(np.asarray(X.columns), self.feature_names_in_):
                raise ValueError('X columns must match the names and order seen at fit')
        y = np.asarray(self.model_.predict(X)).ravel()
        if self.target_scaler_ is not None:
            y = self.target_scaler_.inverse_transform(y.reshape(-1, 1)).ravel()
        return np.maximum(np.expm1(y), 0)


    def get_feature_importance(self) -> dict[str, np.ndarray]:
        """Return available coefficients or model-specific feature importance."""
        check_is_fitted(self, 'model_')
        if hasattr(self.model_, 'coef_'):
            return {'coef': np.asarray(self.model_.coef_)}
        if hasattr(self.model_, 'booster_'):
            return {'gain': self.model_.booster_.feature_importance(importance_type='gain')}
        if hasattr(self.model_, 'get_feature_importance'):
            return {'prediction_values_change': self.model_.get_feature_importance()}
        if hasattr(self.model_, 'feature_importances_'):
            return {'importance': np.asarray(self.model_.feature_importances_)}
        return {}


    def save_state(self, *, path: str) -> None:
        check_is_fitted(self, 'model_')
        with open(path, 'wb') as f:
            pickle.dump(self, f)


    @classmethod
    def load_state(cls, *, path: str) -> 'ModelTrainer':
        with open(path, 'rb') as f:
            trainer = pickle.load(f)
        if not isinstance(trainer, cls):
            raise TypeError(f'Expected {cls.__name__}, got {type(trainer).__name__}')
        check_is_fitted(trainer, 'model_')
        return trainer
