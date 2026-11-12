import os
import pickle
import logging

import pandas as pd
import numpy as np

from typing import Any, cast
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from scipy.stats import norm, loguniform, uniform, randint

from data_transformer import DataTransformer
from model_trainer import ModelTrainer
from dl_model import DLM
from hyperparameter_search import HyperparameterSearch
from shap_explainer import ShapExplainer
from utils import get_metrics, LogUniformInt


class ModelManager:
    def __init__(self, *, config: DictConfig) -> None:
        self.logger = logging.getLogger(__name__)
        self.config = config

        if config.general.is_train:
            self.models = {}
            self.state_checkpoint = {'models': {}}
        else:
            self._load_state()
            self._load_models()


    def fit(self, *, df: pd.DataFrame) -> dict:
        os.makedirs(self.config.paths.checkpoints, exist_ok=True)
        os.makedirs(self.config.paths.models, exist_ok=True)
        self.models = {}
        self.state_checkpoint = {'models': {}}
        X = df.drop('SalePrice', axis=1)
        y = df.SalePrice
        y_bins = pd.qcut(np.log1p(y), q=10, labels=False, duplicates="drop")
        X_test, y_test = None, None
        if self.config.general.is_hold_test:
            X, X_test, y, y_test = train_test_split(
                X,
                y, 
                test_size=self.config.general.test_size, 
                random_state=self.config.general.seed,
                stratify=y_bins
            )
            y_bins = pd.qcut(np.log1p(y), q=10, labels=False, duplicates="drop")

        artifacts: dict[str, Any] = {'best_params': {}, 'search_best_scores': {}}
        metrics = {}
        importances = {}
        shap_importances = {}
        all_oof_predictions = {}
        splitter = StratifiedKFold(
            n_splits=self.config.general.n_splits,
            shuffle=self.config.general.is_shuffle,
            random_state=self.config.general.seed if self.config.general.is_shuffle else None
        )
        folds = list(splitter.split(X=X, y=y_bins))

        for model_name, enabled in self.config.models.to_train.items():
            if not enabled:
                continue
            data_transformer = DataTransformer(model_name=model_name, random_state=self.config.general.seed)
            model_trainer = self._create_trainer(model_name=model_name)

            params = self.config.models.params[model_name]
            if params.get('is_hyperparameter_search', False):
                search_kwargs = cast(
                    dict[str, Any],
                    OmegaConf.to_container(params.hyperparameter_search_kwargs, resolve=True)
                )
                search_kwargs['param_distributions'] = self._make_distributions(search_kwargs['param_distributions'])
                search_kwargs.setdefault('random_state', self.config.general.seed)
                search = HyperparameterSearch(
                    data_transformer=data_transformer,
                    model_trainer=model_trainer,
                    **search_kwargs
                ).fit(X, y)
                model_trainer.estimator.set_params(**search.best_params_)
                artifacts['best_params'][model_name] = search.best_params_
                artifacts['search_best_scores'][model_name] = search.best_score_
                model_path = os.path.join(self.config.paths.models, model_name)
                os.makedirs(model_path, exist_ok=True)
                pd.DataFrame(search.cv_results_).to_csv(os.path.join(model_path, 'search_results.csv'), index=False)
                self.logger.info(f'{model_name} best parameters: {search.best_params_}, CV score: {search.best_score_:.6f}')

            all_oof_predictions[model_name] = self._train_one_model(
                X=X, y=y, model_name=model_name, folds=folds,
                data_transformer=data_transformer, model_trainer=model_trainer,
                metrics=metrics, importances=importances, shap_importances=shap_importances
            )

        columns = {}
        for (model_name, kind), per_fold in importances.items():
            all_folds = pd.concat(per_fold, axis=1)
            columns[f'{model_name}_{kind}_mean'] = all_folds.mean(axis=1)
            columns[f'{model_name}_{kind}_std'] = all_folds.std(axis=1)
        pd.DataFrame(columns).to_csv(self.config.paths.importance)

        if shap_importances:
            columns = {}
            for model_name, per_fold in shap_importances.items():
                folds = pd.concat(per_fold, axis=1)
                columns[f'{model_name}_mean'] = folds.mean(axis=1)
                columns[f'{model_name}_std'] = folds.std(axis=1)
            pd.DataFrame(columns).to_csv(self.config.paths.shap_importance)

        ensemble_oof_predictions = self._blend(all_predictions=all_oof_predictions)
        all_oof_predictions['ensemble'] = ensemble_oof_predictions
        metrics['ensemble'] = {'oof': get_metrics(y_true=np.asarray(y), y_pred=ensemble_oof_predictions)}
        pd.DataFrame(all_oof_predictions, index=X.index).assign(target=y).to_csv(self.config.paths.oof)

        if self.config.general.is_eval_test:
            if X_test is None or y_test is None:
                raise ValueError('X_test or y_test is not defined')
            
            all_predictions = self._predict_per_model(X=X_test)
            predictions = self._blend(all_predictions=all_predictions)

            for model_name, preds in all_predictions.items():
                metrics[model_name]['test'] = get_metrics(y_true=y_test, y_pred=preds)
                
            metrics['ensemble']['test'] = get_metrics(y_true=y_test, y_pred=predictions)
            artifacts['test_predictions'] = predictions

        if self.config.general.is_save_models:
            self._save_state()
        artifacts['metrics'] = metrics
        return artifacts


    def predict(self, *, X: pd.DataFrame) -> np.ndarray:
        predictions = self._blend(all_predictions=self._predict_per_model(X=X))
        return predictions


    def _train_one_model(
            self, *, X: pd.DataFrame, y: pd.Series, model_name: str, folds: list, data_transformer: DataTransformer, model_trainer: ModelTrainer, metrics: dict, importances: dict, shap_importances: dict
        ) -> np.ndarray:
        self.logger.info(f'Start train {model_name}')
        model_checkpoint_path = os.path.join(self.config.paths.models, model_name)
        os.makedirs(model_checkpoint_path, exist_ok=True)

        self.models[model_name] = {}
        self.state_checkpoint['models'][model_name] = {}
        best_epochs = []
        shap_per_fold = []
        shap_data_per_fold = []
        metrics[model_name] = {'per_fold': []}
        oof = np.zeros(len(X))
        val_size = self.config.models.params[model_name].get('val_size', 0)
        use_eval_set = model_name in ('CatBoostRegressor', 'LGBMRegressor', 'XGBRegressor') and val_size > 0

        for i, (train_index, val_index) in tqdm(enumerate(folds), total=len(folds), desc=f'Model: {model_name}'):
            fold_checkpoint_path = os.path.join(model_checkpoint_path, f'fold_{i}')
            os.makedirs(fold_checkpoint_path, exist_ok=True)

            data_transformer_path = os.path.join(fold_checkpoint_path, 'transformer.pickle')
            model_trainer_path = os.path.join(fold_checkpoint_path, 'model.pickle')
            fold_data_transformer = clone(data_transformer)
            fold_data_transformer.set_params(verbose=1 if i == 0 else 0)
            fold_model_trainer = clone(model_trainer)
            X_train = X.iloc[train_index]
            y_train = y.iloc[train_index]
            fit_kwargs = {}
            if use_eval_set:
                X_train, X_eval, y_train, y_eval = train_test_split(
                    X_train, y_train,
                    test_size=val_size,
                    random_state=self.config.general.seed
                )
            X_train = fold_data_transformer.fit_transform(X_train, y_train)
            X_val = fold_data_transformer.transform(X.iloc[val_index])
            if use_eval_set:
                X_eval = fold_data_transformer.transform(X_eval)
                fit_kwargs['eval_set'] = [(X_eval, np.asarray(y_eval))]
            y_train = np.asarray(y_train)
            y_val = np.asarray(y.iloc[val_index])
            fold_model_trainer.fit(X_train, y_train, **fit_kwargs)
            artifacts_fold = fold_model_trainer.artifacts_
            if 'history' in artifacts_fold:
                history = {k: v for k, v in artifacts_fold['history'].items() if v}
                pd.DataFrame(history).to_csv(os.path.join(fold_checkpoint_path, 'history.csv'), index=False)
                best_epochs.append(artifacts_fold['best_epoch'])
            val_predictions = fold_model_trainer.predict(X=X_val)
            val_metrics = get_metrics(y_true=y_val, y_pred=val_predictions)

            self.models[model_name][f'fold_{i}'] = {
                'data_transformer': fold_data_transformer, 
                'model_trainer': fold_model_trainer
            }
            self.state_checkpoint['models'][model_name][f'fold_{i}'] = {
                'data_transformer_path': os.path.abspath(data_transformer_path), 
                'model_trainer_path': os.path.abspath(model_trainer_path)
            }

            metrics[model_name]['per_fold'].append(val_metrics)
            oof[val_index] = val_predictions

            names = fold_data_transformer.get_feature_names()
            for kind, values in fold_model_trainer.get_feature_importance().items():
                importances.setdefault((model_name, kind), []).append(pd.Series(values, index=names))

            if self.config.general.is_shap:
                shap_values = ShapExplainer(
                    predict=fold_model_trainer.predict,
                    random_state=self.config.general.seed
                ).get_shap_values(X_background=X_train, X_explain=X_val)
                grouped = fold_data_transformer.group_by_source_feature(shap_values)
                grouped.index = X.index[val_index]
                shap_per_fold.append(grouped)

                transformed_names = list(fold_data_transformer.get_feature_names())
                transformed_values = pd.DataFrame(
                    np.asarray(X_val),
                    columns=transformed_names,
                    index=X.index[val_index],
                )
                raw_fold = X.iloc[val_index]
                shap_data = pd.DataFrame(index=X.index[val_index])

                for feature in grouped.columns:
                    if feature in raw_fold.columns:
                        shap_data[feature] = raw_fold[feature]
                    elif feature in transformed_values.columns:
                        shap_data[feature] = transformed_values[feature]
                    else:
                        shap_data[feature] = np.nan

                shap_data_per_fold.append(shap_data)

            if self.config.general.is_save_models:
                fold_data_transformer.save_state(path=data_transformer_path)
                fold_model_trainer.save_state(path=model_trainer_path)

        metrics[model_name]['oof'] = get_metrics(y_true=np.asarray(y), y_pred=oof)
        if best_epochs:
            self.logger.info(
                f'{model_name} best epoch by fold: {best_epochs}, '
                f'mean {np.mean(best_epochs):.1f} ± {np.std(best_epochs):.1f}'
            )
        if self.config.general.is_shap:
            oof_shap = pd.concat(shap_per_fold).loc[X.index]
            oof_shap_data = pd.concat(shap_data_per_fold).loc[X.index]
            oof_shap.to_csv(os.path.join(model_checkpoint_path, 'shap_oof.csv'))
            oof_shap_data.to_csv(os.path.join(model_checkpoint_path, 'shap_data_oof.csv'))
            shap_importances[model_name] = [fold.abs().mean() for fold in shap_per_fold]
            
        return oof


    def _predict_per_model(self, *, X: pd.DataFrame) -> dict:
        all_predictions = {}
        for model_name, fold_models in self.models.items():
            all_predictions[model_name] = np.zeros(len(X))

            for models in fold_models.values():
                X_transformed = models['data_transformer'].transform(X=X)
                predictions = models['model_trainer'].predict(X=X_transformed)
                all_predictions[model_name] += predictions / len(fold_models)

        return all_predictions


    def _blend(self, *, all_predictions: dict[str, np.ndarray]) -> np.ndarray:
        weights = {name: value for name, value in self.config.models.to_inference.items() if value > 0}
        blended = np.zeros_like(all_predictions[next(iter(weights))], dtype=float)
        for model_name, weight in weights.items():
            blended += all_predictions[model_name] * weight
        return blended / sum(weights.values())


    def _save_state(self) -> None:
        with open(self.config.paths.manager_checkpoint, "wb") as f:
            pickle.dump(self.state_checkpoint, f)


    def _load_state(self) -> None:
        with open(self.config.paths.manager_checkpoint, 'rb') as f:
            self.state_checkpoint = pickle.load(f)


    def _load_models(self) -> None:
        self.models = {}
        weights = {name: value for name, value in self.config.models.to_inference.items() if value > 0}
        for model_name in weights:
            folds = self.state_checkpoint['models'][model_name]
            self.models[model_name] = {}
            for fold_name, paths in tqdm(folds.items(), desc=f'Model: {model_name}'):
                self.models[model_name][fold_name] = {
                    'data_transformer': DataTransformer.load_state(path=paths['data_transformer_path']),
                    'model_trainer': ModelTrainer.load_state(path=paths['model_trainer_path'])
                }


    def _create_trainer(self, *, model_name: str) -> ModelTrainer:
        classes = {
            'BaseLineLinearRegression': LinearRegression,
            'LinearRegression': LinearRegression,
            'Ridge': Ridge,
            'Lasso': Lasso,
            'ElasticNet': ElasticNet,
            'CatBoostRegressor': CatBoostRegressor,
            'LGBMRegressor': LGBMRegressor,
            'XGBRegressor': XGBRegressor,
            'DLM': DLM
        }
        if model_name not in classes:
            raise ValueError(f'Unknown model: {model_name}')
        params = self.config.models.params[model_name]
        kwargs = cast(dict[str, Any], OmegaConf.to_container(params.kwargs, resolve=True))
        fit_kwargs = cast(
            dict[str, Any],
            OmegaConf.to_container(params.get('fit_kwargs', OmegaConf.create({})), resolve=True)
        )
        return ModelTrainer(
            estimator=classes[model_name](**kwargs),
            is_scale_target=params.get('is_scale_target', False),
            winsorization_value=params.get('winsorization_value'),
            fit_kwargs=fit_kwargs
        )


    @staticmethod
    def _make_distributions(spec: dict) -> dict:
        distributions = {}
        for name, value in spec.items():
            if not isinstance(value, dict):
                distributions[name] = value
                continue
            kind = value['type']
            if kind == 'choice':
                distributions[name] = value['values']
            elif kind == 'loguniform':
                distributions[name] = loguniform(value['a'], value['b'])
            elif kind == 'uniform':
                distributions[name] = uniform(loc=value['a'], scale=value['b'] - value['a'])
            elif kind == 'int':
                distributions[name] = randint(value['a'], value['b'] + 1)
            elif kind == 'norm':
                distributions[name] = norm(loc=value['a'], scale=value['b'])
            elif kind == 'logint':
                distributions[name] = LogUniformInt(value['a'], value['b'])
            else:
                raise ValueError(f'Unknown distribution type: {kind}')
        return distributions
