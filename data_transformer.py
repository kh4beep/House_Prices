import os
import pickle
import logging

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, QuantileTransformer, PowerTransformer, FunctionTransformer, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

from custom_transformers import on_columns, EstimatorImputer, MyTargetEncoder, SafeCategoryTransformer, GroupStatImputer


class DataTransformer:
    """Per-model preprocessing pipeline fitted on train rows only; includes target-derived features."""
    def __init__(self, *, config, path: str, model_name: str, verbose: int = 0):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.path = path
        self.model_name = model_name
        self.verbose = verbose
        self.is_as_frame = model_name in ['CatBoostRegressor', 'LGBMRegressor', 'XGBRegressor']
        self.state_checkpoint = {}  
        self.pipeline = self._build_pipeline()


    def _build_pipeline(self):
        """Build pipeline for each model"""
        seed = self.config.general.seed

        if self.model_name == 'BaseLineLinearRegression':
            pipeline = Pipeline([
                ('impute_median', on_columns(SimpleImputer(strategy='median'), ['GarageArea'])),
                ('scale', on_columns(
                    StandardScaler(),
                    [
                        'OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt'
                    ]
                )),
                ('drop', on_columns('drop', [
                    'MSSubClass', 'MSZoning', 'LotFrontage', 'Alley', 'LotShape', 'LandContour',
                    'LotConfig', 'LandSlope', 'Neighborhood', 'BldgType', 'HouseStyle',
                    'OverallCond', 'YearRemodAdd', 'RoofStyle',
                    'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual',
                    'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                    'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF',
                    'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
                    'BsmtFullBath', 'BsmtHalfBath', 'HalfBath',
                    'KitchenAbvGr', 'KitchenQual', 'Functional', 'Fireplaces',
                    'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars',
                    'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF',
                    'OpenPorchSF', 'EnclosedPorch', 'ScreenPorch', 'PoolQC', 'Fence', 'MiscVal',
                    'SaleType', 'SaleCondition', 'YrSold', 'MoSold', 'LotArea', 'BedroomAbvGr',
                    'Condition_Artery', 'Condition_RR', 'Condition_Pos', 'MiscFeature_Shed',
                    'MiscFeature_other', 'MoSold_sin', 'MoSold_cos'
                ]))
            ])
        elif (self.model_name in ('LinearRegression', 'Ridge', 'Lasso', 'ElasticNet', 'DLM')):
            pipeline = Pipeline([
                ('impute_MSZoning', on_columns(GroupStatImputer(target_col='MSZoning', group_cols=['Neighborhood'], method='mode'), ['MSZoning', 'Neighborhood'])),
                ('impute_Exterior1st', on_columns(GroupStatImputer(target_col='Exterior1st', group_cols=['Neighborhood'], method='mode'), ['Exterior1st', 'Neighborhood'])),
                ('impute_Exterior2nd', on_columns(GroupStatImputer(target_col='Exterior2nd', group_cols=['Neighborhood'], method='mode'), ['Exterior2nd', 'Neighborhood'])),
                ('impute_LotFrontage', on_columns(GroupStatImputer(target_col='LotFrontage', group_cols=['Neighborhood'], method='median'), ['LotFrontage', 'Neighborhood'])),
                ('impute_most_frequent', on_columns(SimpleImputer(strategy='most_frequent'), ['Electrical'])),
                ('impute_most_frequent_ord', on_columns(SimpleImputer(strategy='most_frequent'), ['BsmtCond', 'GarageCond', 'GarageFinish'])),
                ('impute_median', on_columns(SimpleImputer(strategy='median'), ['GarageArea', 'GarageCars'])),
                ('onehot', on_columns(
                    OneHotEncoder(drop='first', min_frequency=5, handle_unknown='infrequent_if_exist', sparse_output=False),
                    [
                        'MSSubClass', 'MSZoning', 'Alley', 'LandContour', 'LotConfig', 'LandSlope',
                        'Neighborhood', 'BldgType', 'HouseStyle', 'RoofStyle', 'Exterior1st',
                        'Exterior2nd', 'MasVnrType', 'Foundation', 'CentralAir', 'Electrical',
                        'GarageType', 'PavedDrive', 'SaleType', 'SaleCondition'
                    ]
                )),
                ('transform_log', on_columns(
                    FunctionTransformer(np.log1p),
                    [
                        'LotFrontage', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', '2ndFlrSF',
                        'EnclosedPorch', 'ScreenPorch'
                    ]
                )),
                ('transform_yeo-johnson', on_columns(
                    PowerTransformer(method='yeo-johnson'),
                    ['BsmtUnfSF', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF']
                )),
                ('transform_box-cox', on_columns(
                    PowerTransformer(method='box-cox'),
                    ['1stFlrSF', 'GrLivArea']
                )),
                ('transform_quantile', on_columns(
                    QuantileTransformer(n_quantiles=100, output_distribution='normal', random_state=seed),
                    ['LotArea', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', 'GarageYrBlt']
                )),
                ('scale', on_columns(
                    StandardScaler(),
                    [
                        'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',
                        'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                        'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'BsmtFullBath',
                        'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
                        'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea',
                        'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'ScreenPorch', 'YrSold'
                    ]
                ))
            ])
        elif self.model_name == 'CatBoostRegressor':
            pipeline = Pipeline([
                ('impute_MSZoning', on_columns(GroupStatImputer(target_col='MSZoning', group_cols=['Neighborhood'], method='mode'), ['MSZoning', 'Neighborhood'])),
                ('impute_Exterior1st', on_columns(GroupStatImputer(target_col='Exterior1st', group_cols=['Neighborhood'], method='mode'), ['Exterior1st', 'Neighborhood'])),
                ('impute_Exterior2nd', on_columns(GroupStatImputer(target_col='Exterior2nd', group_cols=['Neighborhood'], method='mode'), ['Exterior2nd', 'Neighborhood'])),
                ('impute_LotFrontage', on_columns(GroupStatImputer(target_col='LotFrontage', group_cols=['Neighborhood'], method='median'), ['LotFrontage', 'Neighborhood'])),
                ('impute_most_frequent', on_columns(SimpleImputer(strategy='most_frequent'), ['Electrical'])),
                ('impute_median', on_columns(SimpleImputer(strategy='median'), ['GarageArea', 'GarageCars', 'BsmtCond'])),
            ])
        elif self.model_name in ('LGBMRegressor', 'XGBRegressor'):
            pipeline = Pipeline([
                ('impute_MSZoning', on_columns(GroupStatImputer(target_col='MSZoning', group_cols=['Neighborhood'], method='mode'), ['MSZoning', 'Neighborhood'])),
                ('impute_Exterior1st', on_columns(GroupStatImputer(target_col='Exterior1st', group_cols=['Neighborhood'], method='mode'), ['Exterior1st', 'Neighborhood'])),
                ('impute_Exterior2nd', on_columns(GroupStatImputer(target_col='Exterior2nd', group_cols=['Neighborhood'], method='mode'), ['Exterior2nd', 'Neighborhood'])),
                ('impute_LotFrontage', on_columns(GroupStatImputer(target_col='LotFrontage', group_cols=['Neighborhood'], method='median'), ['LotFrontage', 'Neighborhood'])),
                ('impute_most_frequent', on_columns(SimpleImputer(strategy='most_frequent'), ['Electrical'])),
                ('impute_median', on_columns(SimpleImputer(strategy='median'), ['GarageArea', 'GarageCars', 'BsmtCond'])),
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
            self.state_checkpoint['feature_names'] = columns
        elif columns != self.state_checkpoint['feature_names']:
            raise ValueError('Output columns differ from those seen at fit')

        numeric = X_transformed.select_dtypes('number')
        values = numeric.to_numpy(dtype=float)
        bad = np.isinf(values) if self.is_as_frame else ~np.isfinite(values)
        bad_columns = numeric.columns[bad.any(axis=0)].tolist()
        if bad_columns:
            raise ValueError(f'NaN or inf after transform in: {bad_columns}')

        if is_fit:
            self.state_checkpoint['train_mean'] = numeric.mean()
            self.state_checkpoint['train_std'] = numeric.std()
        elif self.verbose:
            train_std = self.state_checkpoint['train_std'].replace(0, np.nan)
            shift = ((numeric.mean() - self.state_checkpoint['train_mean']) / train_std).abs()
            drifted = shift[shift > 0.5].round(2)
            if not drifted.empty:
                self.logger.warning(f'Mean shift vs train, in train std: {drifted.to_dict()}')


    def _as_frame(self, X_transformed) -> pd.DataFrame:
        if not isinstance(X_transformed, pd.DataFrame):
            raise TypeError(f'Pipeline must return DataFrame, got {type(X_transformed).__name__}')
        return X_transformed


    def fit(self, *, X: pd.DataFrame, y: pd.Series | None = None) -> 'DataTransformer':
        self.fit_transform(X=X, y=y) # fit_transform, not fit: MyTargetEncoder cross-fits train rows only inside fit_transform
        return self


    def transform(self, *, X: pd.DataFrame) -> pd.DataFrame | np.ndarray:
        X_transformed = self._as_frame(self.pipeline.transform(X))
        self._validate_output(X=X, X_transformed=X_transformed, is_fit=False)
        return X_transformed if self.is_as_frame else np.asarray(X_transformed, dtype=float)


    def fit_transform(self, *, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame | np.ndarray:
        self.state_checkpoint['input_columns'] = list(X.columns)
        X_transformed = self._as_frame(self.pipeline.fit_transform(X, y))
        self._validate_output(X=X, X_transformed=X_transformed, is_fit=True)
        return X_transformed if self.is_as_frame else np.asarray(X_transformed, dtype=float)


    def get_feature_names(self) -> np.ndarray:
        return self.pipeline.get_feature_names_out()


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
        names = self.state_checkpoint['feature_names']
        inputs = self.state_checkpoint['input_columns']
        sources = [self._source_feature(name, inputs) for name in names]
        frame = pd.DataFrame(np.asarray(values), columns=names)
        return frame.T.groupby(sources, sort=False).sum().T


    def save_state(self) -> None:
        os.makedirs(self.path, exist_ok=True)
        
        with open(os.path.join(self.path, 'state_checkpoint.pickle'), 'wb') as f:
            pickle.dump(self.state_checkpoint, f)

        with open(os.path.join(self.path, 'pipeline.pickle'), 'wb') as f:
            pickle.dump(self.pipeline, f)


    def load_state(self) -> None:
        with open(os.path.join(self.path, 'state_checkpoint.pickle'), 'rb') as f:
            self.state_checkpoint = pickle.load(f)

        with open(os.path.join(self.path, 'pipeline.pickle'), 'rb') as f:
            self.pipeline = pickle.load(f)
