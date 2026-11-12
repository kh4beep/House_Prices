import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from data_transformer import DataTransformer
from model_trainer import ModelTrainer


class HyperparameterSearch:
    def __init__(
        self,
        *,
        data_transformer: DataTransformer,
        model_trainer: ModelTrainer,
        param_distributions: dict,
        n_iter: int = 20,
        cv=5,
        scoring: str = 'neg_root_mean_squared_log_error',
        random_state: int = 42,
        n_jobs: int = 1,
        verbose: int = 0,
    ):
        self.data_transformer = data_transformer
        self.model_trainer = model_trainer
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose


    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray, *, groups=None) -> 'HyperparameterSearch':
        if 'eval_set' in (self.model_trainer.fit_kwargs or {}):
            raise ValueError('A fixed eval_set is not supported during cross-validation search')
        pipeline = Pipeline([
            ('data', clone(self.data_transformer)),
            ('model', clone(self.model_trainer))
        ])
        distributions = {
            f'model__estimator__{name}': values
            for name, values in self.param_distributions.items()
        }
        cv = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state) if isinstance(self.cv, int) else self.cv
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=distributions,
            n_iter=self.n_iter,
            scoring=self.scoring,
            cv=cv,
            refit=False,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            error_score='raise'
        )
        search.fit(X, y, groups=groups)
        self.best_params_ = {
            name.removeprefix('model__estimator__'): value
            for name, value in search.best_params_.items()
        }
        self.best_score_ = float(search.best_score_)
        self.best_index_ = int(search.best_index_)
        self.cv_results_ = search.cv_results_
        self.n_splits_ = search.n_splits_
        return self
