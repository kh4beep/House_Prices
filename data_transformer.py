import pickle
import logging

import numpy as np
import pandas as pd

from typing import Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, QuantileTransformer, PowerTransformer, FunctionTransformer
from sklearn.compose import ColumnTransformer

from custom_transformers import on_columns, SafeCategoryTransformer, GroupStatImputer, CombinedFeatureCreator, GroupByFeatureCreator, MaxValueClipper


class DataTransformer(TransformerMixin, BaseEstimator):
    """Per-model preprocessing pipeline fitted on train rows only; includes target-derived features."""
    def __init__(self, *, model_name: str, random_state: int = 42, verbose: int = 0):
        self.model_name = model_name
        self.random_state = random_state
        self.verbose = verbose


    def _build_pipeline(self):
        """Build pipeline for each model"""
        seed = self.random_state

        if self.model_name == 'BaseLineLinearRegression':
            pipeline = Pipeline([
                ('impute_median', on_columns(SimpleImputer(strategy='median'), ['GarageArea'])),
                ('scale', on_columns(
                    StandardScaler(),
                    [
                        'OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt'
                    ]
                )),
                ('select', ColumnTransformer(
                    [('features', 'passthrough', [
                        'OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF',
                        'FullBath', 'TotRmsAbvGrd', 'YearBuilt'
                    ])],
                    remainder='drop',
                    verbose_feature_names_out=False
                ))
            ])
        elif (self.model_name in ('LinearRegression', 'Ridge', 'Lasso', 'ElasticNet', 'DLM')):
            pipeline = Pipeline([
                ('impute_median', on_columns(
                    SimpleImputer(strategy='median'),
                    ['GarageArea', 'GarageCars', 'GarageFinish']
                )),
                ('impute_most_frequent', on_columns(
                    SimpleImputer(strategy='most_frequent'),
                    ['Electrical', 'MSZoning']
                )),
                ('impute_LotFrontage', GroupStatImputer(target_col='LotFrontage', group_cols=['Neighborhood'], method='median')),
                ('new_HouseGarageSF', CombinedFeatureCreator(features=['GrLivArea', 'GarageArea'], feature_name='HouseGarageSF')),
                ('new_TotalPropertySF', CombinedFeatureCreator(features=['GrLivArea', 'TotalBsmtSF', 'GarageArea'], feature_name='TotalPropertySF')),
                ('new_GarageHouseRatio', CombinedFeatureCreator(features=['GarageArea', 'GrLivArea'], feature_name='GarageHouseRatio', operation='div')),
                ('new_QualityGarageSF', CombinedFeatureCreator(features=['GarageQual', 'GarageArea'], feature_name='QualityGarageSF', operation='product')),
                ('new_NeighborhoodAvgLotArea', GroupByFeatureCreator(group_cols=['Neighborhood'], source_col='LotArea', feature_name='NeighborhoodAvgLotArea')),
                ('new_NeighborhoodAvgGrLivArea', GroupByFeatureCreator(group_cols=['Neighborhood'], source_col='GrLivArea', feature_name='NeighborhoodAvgGrLivArea')),
                ('new_NeighborhoodAvgOverallQual', GroupByFeatureCreator(group_cols=['Neighborhood'], source_col='OverallQual', feature_name='NeighborhoodAvgOverallQual')),
                ('onehot', on_columns(
                    OneHotEncoder(drop='first', min_frequency=5, handle_unknown='infrequent_if_exist', sparse_output=False),
                    [
                        'Alley', 'BldgType', 'CentralAir', 'Electrical', 'Exterior1st', 'Exterior2nd',
                        'Foundation', 'GarageType', 'HouseStyle', 'LandContour', 'LandSlope',
                        'LotConfig', 'MasVnrType', 'MSSubClass', 'MSZoning', 'Neighborhood',
                        'PavedDrive', 'RoofStyle', 'SaleCondition', 'SaleType'
                    ]
                )),
                ('clip', MaxValueClipper(percentile=99.5, features=[
                    '1stFlrSF', 'BsmtFinSF1', 'BsmtUnfSF', 'GarageArea', 'GrLivArea', 'LotArea',
                    'OpenPorchSF', 'TotalBsmtSF'
                ])),
                ('transform_log', on_columns(
                    FunctionTransformer(func=np.log1p),
                    [
                        '2ndFlrSF', 'BsmtFinSF2', 'EnclosedPorch', 'LotArea', 'LotFrontage',
                        'MasVnrArea', 'OpenPorchSF', 'ScreenPorch', 'WoodDeckSF',
                        'YearsToRemod', 'SFPerRoom', 'QualityBasementSF'
                    ]
                )),
                ('transform_yeo-johnson', on_columns(
                    PowerTransformer(method='yeo-johnson'),
                    [
                        'BsmtFinSF1', 'BsmtUnfSF', 'GarageArea', 'TotalBsmtSF', 'TotalPorchSF',
                        'LotBuildingRatio', 'QualityGarageSF', 'GarageHouseAgeDiff'
                    ]
                )),
                ('transform_box-cox', on_columns(
                    PowerTransformer(method='box-cox'),
                    [
                        '1stFlrSF', 'GrLivArea', 'QualitySF', 'QualityTotalSF', 'TotalHouseSF',
                        'HouseGarageSF', 'TotalPropertySF'
                    ]
                )),
                ('transform_quantile', on_columns(
                    QuantileTransformer(n_quantiles=100, output_distribution='normal', random_state=seed),
                    ['GarageYrBlt', 'YearBuilt', 'YearRemodAdd', 'HouseAge', 'GarageAge', 'YearsSinceRemod']
                )),
                ('scale', on_columns(
                    StandardScaler(),
                    [
                        '2ndFlrSF', 'BedroomAbvGr', 'BsmtCond', 'BsmtExposure', 'BsmtFinSF2',
                        'BsmtFinType1', 'BsmtFinType2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtQual',
                        'EnclosedPorch', 'ExterCond', 'ExterQual', 'Fence', 'FireplaceQu',
                        'Fireplaces', 'FullBath', 'Functional', 'GarageCars', 'GarageCond',
                        'GarageFinish', 'GarageQual', 'GarageYrBlt', 'HalfBath', 'HeatingQC',
                        'KitchenAbvGr', 'KitchenQual', 'LotArea', 'LotFrontage', 'LotShape',
                        'MasVnrArea', 'MiscVal', 'MoSold', 'OpenPorchSF', 'OverallCond',
                        'OverallQual', 'PoolQC', 'ScreenPorch', 'TotRmsAbvGrd', 'WoodDeckSF',
                        'YearBuilt', 'YearRemodAdd', 'YrSold', 'DateSold', 'MoSold_sin', 'MoSold_cos',
                        'HouseAge', 'GarageAge', 'GarageHouseAgeDiff', 'YearsSinceRemod',
                        'YearsToRemod', 'TotalBath', 'SFPerRoom', 'OtherRmsAbvGrd', 'QualityBasementSF',
                        'BsmtFinishedRatio', 'GarageHouseRatio', 'NeighborhoodAvgLotArea',
                        'NeighborhoodAvgGrLivArea', 'NeighborhoodAvgOverallQual'
                    ]
                )),
            ])
        elif self.model_name == 'CatBoostRegressor':
            pipeline = Pipeline([
                ('impute_median', on_columns(
                    SimpleImputer(strategy='median'),
                    ['GarageArea', 'GarageCars', 'GarageFinish']
                )),
                ('impute_most_frequent', on_columns(
                    SimpleImputer(strategy='most_frequent'),
                    ['Electrical', 'MSZoning']
                )),
                ('impute_LotFrontage', GroupStatImputer(target_col='LotFrontage', group_cols=['Neighborhood'], method='median')),
                ('new_HouseGarageSF', CombinedFeatureCreator(features=['GrLivArea', 'GarageArea'], feature_name='HouseGarageSF')),
                ('new_TotalPropertySF', CombinedFeatureCreator(features=['GrLivArea', 'TotalBsmtSF', 'GarageArea'], feature_name='TotalPropertySF')),
                ('new_GarageHouseRatio', CombinedFeatureCreator(features=['GarageArea', 'GrLivArea'], feature_name='GarageHouseRatio', operation='div')),
                ('new_QualityGarageSF', CombinedFeatureCreator(features=['GarageQual', 'GarageArea'], feature_name='QualityGarageSF', operation='product')),
                ('new_NeighborhoodAvgLotArea', GroupByFeatureCreator(group_cols=['Neighborhood'], source_col='LotArea', feature_name='NeighborhoodAvgLotArea')),
                ('new_NeighborhoodAvgGrLivArea', GroupByFeatureCreator(group_cols=['Neighborhood'], source_col='GrLivArea', feature_name='NeighborhoodAvgGrLivArea')),
                ('new_NeighborhoodAvgOverallQual', GroupByFeatureCreator(group_cols=['Neighborhood'], source_col='OverallQual', feature_name='NeighborhoodAvgOverallQual')),
            ])
        elif self.model_name in ('LGBMRegressor', 'XGBRegressor'):
            pipeline = Pipeline([
                ('impute_median', on_columns(
                    SimpleImputer(strategy='median'),
                    ['GarageArea', 'GarageCars', 'GarageFinish']
                )),
                ('impute_most_frequent', on_columns(
                    SimpleImputer(strategy='most_frequent'),
                    ['Electrical', 'MSZoning']
                )),
                ('impute_LotFrontage', GroupStatImputer(target_col='LotFrontage', group_cols=['Neighborhood'], method='median')),
                ('new_HouseGarageSF', CombinedFeatureCreator(features=['GrLivArea', 'GarageArea'], feature_name='HouseGarageSF')),
                ('new_TotalPropertySF', CombinedFeatureCreator(features=['GrLivArea', 'TotalBsmtSF', 'GarageArea'], feature_name='TotalPropertySF')),
                ('new_GarageHouseRatio', CombinedFeatureCreator(features=['GarageArea', 'GrLivArea'], feature_name='GarageHouseRatio', operation='div')),
                ('new_QualityGarageSF', CombinedFeatureCreator(features=['GarageQual', 'GarageArea'], feature_name='QualityGarageSF', operation='product')),
                ('new_NeighborhoodAvgLotArea', GroupByFeatureCreator(group_cols=['Neighborhood'], source_col='LotArea', feature_name='NeighborhoodAvgLotArea')),
                ('new_NeighborhoodAvgGrLivArea', GroupByFeatureCreator(group_cols=['Neighborhood'], source_col='GrLivArea', feature_name='NeighborhoodAvgGrLivArea')),
                ('new_NeighborhoodAvgOverallQual', GroupByFeatureCreator(group_cols=['Neighborhood'], source_col='OverallQual', feature_name='NeighborhoodAvgOverallQual')),
                ('clean_unknown', on_columns(SafeCategoryTransformer(), 
                    [
                        'MSSubClass', 'MSZoning', 'Alley', 'LandContour', 'LotConfig', 'LandSlope',
                        'Neighborhood', 'BldgType', 'HouseStyle', 'RoofStyle', 'Exterior1st',
                        'Exterior2nd', 'MasVnrType', 'Foundation', 'CentralAir', 'Electrical',
                        'GarageType', 'PavedDrive', 'SaleType', 'SaleCondition'
                    ]
                ))
            ])
        else:
            raise ValueError(f'Pipeline for {self.model_name} is not defined')

        pipeline.set_output(transform='pandas')
        return pipeline


    def _validate_output(self, *, X: pd.DataFrame, X_transformed: pd.DataFrame, is_fit: bool) -> None:
        """Catch silent preprocessing bugs, i.e. those that would not crash model training.

        - rows dropped or reordered: features no longer aligned with y
        - columns renamed or reordered vs fit: a model fed an ndarray reads the wrong features
        - NaN/inf in numeric features: tree models and DLM train on them without an error
        - mean shift vs train (warning only): fit and transform behave inconsistently
        """
        if not X_transformed.index.equals(X.index):
            raise ValueError('Rows were dropped or reordered during transform')

        columns = list(X_transformed.columns)
        if is_fit:
            self.state_checkpoint_['feature_names'] = columns
        elif columns != self.state_checkpoint_['feature_names']:
            raise ValueError('Output columns differ from those seen at fit')

        numeric = X_transformed.select_dtypes('number')
        values = numeric.to_numpy(dtype=float)
        bad = np.isinf(values) if self.model_name in ('LGBMRegressor', 'XGBRegressor') else ~np.isfinite(values)
        bad_columns = numeric.columns[bad.any(axis=0)].tolist()
        if bad_columns:
            raise ValueError(f'NaN or inf after transform in: {bad_columns}')

        if is_fit:
            self.state_checkpoint_['train_mean'] = numeric.mean()
            self.state_checkpoint_['train_std'] = numeric.std()
        elif self.verbose:
            train_std = self.state_checkpoint_['train_std'].replace(0, np.nan)
            shift = ((numeric.mean() - self.state_checkpoint_['train_mean']) / train_std).abs()
            drifted = shift[shift > 0.5].round(2)
            if not drifted.empty:
                logging.getLogger(__name__).warning(f'Mean shift vs train, in train std: {drifted.to_dict()}')


    def _as_frame(self, X_transformed) -> pd.DataFrame:
        if not isinstance(X_transformed, pd.DataFrame):
            raise TypeError(f'Pipeline must return DataFrame, got {type(X_transformed).__name__}')
        return X_transformed


    def fit(self, X: pd.DataFrame, y=None) -> 'DataTransformer':
        self.fit_transform(X, y)
        return self


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, 'pipeline_')
        if not np.array_equal(np.asarray(X.columns), self.feature_names_in_):
            raise ValueError('X columns must match the names and order seen at fit')
        X_transformed = self._as_frame(self.pipeline_.transform(X))
        self._validate_output(X=X, X_transformed=X_transformed, is_fit=False)
        return X_transformed


    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        pipeline = self._build_pipeline()
        X_transformed = self._as_frame(pipeline.fit_transform(X, y))
        self.state_checkpoint_: dict[str, Any] = {'input_columns': list(X.columns)}
        self._validate_output(X=X, X_transformed=X_transformed, is_fit=True)
        self.pipeline_ = pipeline
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return X_transformed


    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, 'pipeline_')
        if input_features is not None and not np.array_equal(input_features, self.feature_names_in_):
            raise ValueError('input_features must match feature_names_in_')
        return np.asarray(self.state_checkpoint_['feature_names'], dtype=object)


    def get_feature_names(self) -> np.ndarray:
        return self.get_feature_names_out()


    @staticmethod
    def _source_feature(name: str, input_columns: list[str]) -> str:
        if name in input_columns:
            return name
        prefixes = [c for c in input_columns if name.startswith(f'{c}_')]
        return max(prefixes, key=len) if prefixes else name


    def group_by_source_feature(self, values: np.ndarray) -> pd.DataFrame:
        """
        Sum per-column values (e.g. SHAP, rows x transformed columns) into source features.
        One-hot columns collapse back into their input feature; engineered columns
        have no source among the inputs and stay as they are.
        Pass raw per-row values and aggregate afterwards: summing is exact for SHAP
        because it is additive, while summing mean |SHAP| would overstate one-hot features.
        """
        check_is_fitted(self, 'pipeline_')
        names = self.state_checkpoint_['feature_names']
        inputs = self.state_checkpoint_['input_columns']
        sources = [self._source_feature(name, inputs) for name in names]
        frame = pd.DataFrame(np.asarray(values), columns=names)
        return frame.T.groupby(sources, sort=False).sum().T


    def save_state(self, *, path: str) -> None:
        check_is_fitted(self, 'pipeline_')
        with open(path, 'wb') as f:
            pickle.dump(self, f)


    @classmethod
    def load_state(cls, *, path: str) -> 'DataTransformer':
        with open(path, 'rb') as f:
            transformer = pickle.load(f)
        if not isinstance(transformer, cls):
            raise TypeError(f'Expected {cls.__name__}, got {type(transformer).__name__}')
        check_is_fitted(transformer, 'pipeline_')
        return transformer
