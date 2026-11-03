import random
import sys
import torch
import logging
import numpy as np
import pandas as pd

from scipy.stats import loguniform
from sklearn.metrics import mean_squared_log_error, r2_score, root_mean_squared_error


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    denominator = np.abs(y_true) + np.abs(y_pred)

    return 100 * np.mean(
        np.divide(
            2 * np.abs(y_true - y_pred),
            denominator,
            out=np.zeros_like(denominator, dtype=float),
            where=denominator != 0,
        )
    )


def set_seed(*, seed: int):
    """Seed Python, NumPy and torch RNGs and make cuDNN deterministic."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def setup_logger(*, path: str) -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.handlers.clear()

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s'
    )

    file_handler = logging.FileHandler(path, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_metrics(*, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Accuracy and F1 at the threshold, ROC AUC on probabilities."""
    y_true = np.asarray(y_true)
    y_pred = np.maximum(np.asarray(y_pred), 0)
    result = {}
    result['r2'] = float(r2_score(y_true, y_pred))
    result['smape'] = float(symmetric_mean_absolute_percentage_error(y_true, y_pred))
    result['rmsle'] = float(root_mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))
    return result


def create_submission(*, index: np.ndarray, predictions: np.ndarray, path: str) -> dict:
    artifacts = {}
    submission = pd.DataFrame({
        'Id': index,
        'SalePrice': predictions
    })
    submission.to_csv(path, index=False)
    artifacts['shape'] = submission.shape
    artifacts['value_counts'] = submission['SalePrice'].value_counts()
    return artifacts


def create_metrics_tables(*, metrics: dict) -> str:
    """
    Format metrics as one table per metric.
    Columns: oof_mean/oof_std over validation folds, oof on all OOF predictions,
    test on the hold-out (NaN if not evaluated).
    """
    rows = []
    for model, m in metrics.items():
        per_fold = pd.DataFrame(m['per_fold']) if 'per_fold' in m else pd.DataFrame()
        for name in m['oof']:
            rows.append({
                'model': model, 'metric': name,
                'oof_mean': per_fold[name].mean() if name in per_fold else np.nan,
                'oof_std': per_fold[name].std() if name in per_fold else np.nan,
                'oof': m['oof'][name],
                'test': m.get('test', {}).get(name, np.nan),
            })
    df = pd.DataFrame(rows).set_index(['metric', 'model']).astype(float).round(4)
    return '\n' + '\n\n'.join(f'{metric}:\n{df.loc[metric].to_string()}' for metric in df.index.unique('metric'))


class LogUniformInt:
    """
    Integers in [a, b] inclusive, uniform on the log scale; a scipy-like rvs for RandomizedSearchCV.
    Samples floor(loguniform(a, b + 1)), so both bounds get their full share. Requires a >= 1.
    """
    def __init__(self, a: int, b: int):
        self.a, self.b = a, b
        self._dist = loguniform(a, b + 1)

    def rvs(self, size=None, random_state=None):
        return np.floor(self._dist.rvs(size=size, random_state=random_state)).astype(int)
