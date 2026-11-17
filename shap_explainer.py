from collections.abc import Callable

import numpy as np
import pandas as pd
import shap


class ShapExplainer:
    """Explain a prediction callable using independent masking and a sampled background."""
    def __init__(self, *, predict: Callable, random_state: int = 42, background_size: int = 100):
        if background_size < 1:
            raise ValueError('background_size must be positive')
        self.predict = predict
        self.random_state = random_state
        self.background_size = background_size


    def get_shap_values(self, *, X_background: pd.DataFrame | np.ndarray, X_explain: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return row-by-feature contributions in the prediction callable's output units."""
        if isinstance(X_explain, pd.DataFrame) and isinstance(X_background, pd.DataFrame):
            encode, decode = self._category_codec(X_background, X_explain)
            background, explain = encode(X_background), encode(X_explain)

            def f(A: np.ndarray) -> np.ndarray:
                return self.predict(decode(A))
        else:
            background, explain = np.asarray(X_background, dtype=float), np.asarray(X_explain, dtype=float)

            def f(A: np.ndarray) -> np.ndarray:
                return self.predict(A)

        background = self._subsample(background, n=self.background_size)
        masker = shap.maskers.Independent(background, max_samples=len(background))
        explainer = shap.Explainer(f, masker, seed=self.random_state)
        explanation = explainer(explain)
        if not isinstance(explanation, shap.Explanation):
            raise TypeError(f'Expected a single Explanation, got {type(explanation).__name__}')
        return np.asarray(explanation.values)


    def _subsample(self, A: np.ndarray, *, n: int) -> np.ndarray:
        if len(A) <= n:
            return A
        idx = np.random.default_rng(self.random_state).choice(len(A), size=n, replace=False)
        return A[idx]


    @staticmethod
    def _category_codec(X_background: pd.DataFrame, X_explain: pd.DataFrame):
        """Encode string columns as integer codes for the masker and decode them back to the original frame."""
        columns, dtypes = X_explain.columns, X_explain.dtypes
        categorical = [c for c in columns if not pd.api.types.is_numeric_dtype(X_explain[c])]
        categories = {c: pd.Index(pd.concat([X_background[c], X_explain[c]]).unique()) for c in categorical}

        def encode(X: pd.DataFrame) -> np.ndarray:
            X = X.copy()
            for c in categorical:
                X[c] = categories[c].get_indexer(pd.Index(X[c]))
            return X.to_numpy(dtype=float)

        def decode(A: np.ndarray) -> pd.DataFrame:
            frame = pd.DataFrame(A, columns=columns)
            for c in categorical:
                frame[c] = categories[c].to_numpy()[frame[c].to_numpy(dtype=int)]
            return frame.astype(dtypes.to_dict())

        return encode, decode
