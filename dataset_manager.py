import pickle
import logging

import numpy as np
import pandas as pd

from omegaconf import DictConfig

class DatasetManager:
    """
    Build row-wise features and cache the resulting dataset.
    Only transformations that use a single row belong here
    """
    def __init__(self, *, config: DictConfig) -> None:
        self.logger = logging.getLogger(__name__)
        self.config = config

        if config.general.is_train and config.general.is_rebuild_dataset:
            self.dataset_checkpoint = {}
        else:
            self._load_dataset_checkpoint()


    def get_dataset(self, *, df: pd.DataFrame) -> pd.DataFrame:
        """Return the feature dataset: rebuilt or loaded from cache depending on config."""
        if not self.config.general.is_train:
            df = self._build_dataset(df=df)
        elif self.config.general.is_rebuild_dataset:
            df = self._build_dataset(df=df)
            self._save_dataset(df=df)
            self._save_dataset_checkpoint()
        else:
            df = self._load_dataset()

        return df


    def _save_dataset_checkpoint(self) -> None:
        with open(self.config.paths.dataset_checkpoint, "wb") as f:
            pickle.dump(self.dataset_checkpoint, f)


    def _load_dataset_checkpoint(self) -> None:
        with open(self.config.paths.dataset_checkpoint, 'rb') as f:
            self.dataset_checkpoint = pickle.load(f)


    def _save_dataset(self, *, df: pd.DataFrame) -> None:
        df.to_csv(self.config.paths.modified_dataset, index=True)


    def _load_dataset(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.paths.modified_dataset, index_col=self.config.general.index_col)
        df = df.astype(self.dataset_checkpoint['dtypes'])
        return df    

    
    def _build_dataset(self, *, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        self.logger.info(f'Initial shape: {df.shape}')

        bsmt_cat = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
        bsmt_num = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'BsmtFullBath', 'BsmtHalfBath']
        df.loc[(df['TotalBsmtSF'].isna()) & (df['BsmtFinType1'].isna()), 'TotalBsmtSF'] = 0.0
        no_bsmt = df['TotalBsmtSF'] == 0.0
        df.loc[no_bsmt, bsmt_cat] = 'No_Basement'
        df.loc[no_bsmt, bsmt_num] = df.loc[no_bsmt, bsmt_num].fillna(0)
        df['BsmtExposure'] = df['BsmtExposure'].fillna('No')
        mask = df['BsmtFinType2'].isna() & df['BsmtFinType1'].notna()
        df.loc[mask, 'BsmtFinType2'] = df.loc[mask, 'BsmtFinType1']

        no_garage = df['GarageType'].isna()
        df.loc[no_garage, ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']] = 'No_Garage'
        df.loc[no_garage, ['GarageYrBlt', 'GarageCars', 'GarageArea']] = 0
        mask = df['GarageYrBlt'].isna() & df['YearBuilt'].notna()
        df.loc[mask, 'GarageYrBlt'] = df.loc[mask, 'YearBuilt']

        qual_from_overall = pd.cut(
            df['OverallQual'], bins=[0, 2, 4, 6, 8, 10], labels=['Po', 'Fa', 'TA', 'Gd', 'Ex']
        ).astype(object)
        df['BsmtQual'] = df['BsmtQual'].fillna(qual_from_overall)
        df['GarageQual'] = df['GarageQual'].fillna(qual_from_overall)
        df['KitchenQual'] = df['KitchenQual'].fillna(qual_from_overall)

        df['Alley'] = df['Alley'].fillna('No_Alley')
        df['MasVnrType'] = df['MasVnrType'].fillna('No_Masonry')
        df['MasVnrArea'] = df['MasVnrArea'].fillna(0.0)
        df['Functional'] = df['Functional'].fillna('Typ')
        df['FireplaceQu'] = df['FireplaceQu'].fillna('No_Fireplace')
        df['PoolQC'] = df['PoolQC'].fillna('No_Pool')
        df['Fence'] = df['Fence'].fillna('No_Fence')
        df['MiscFeature'] = df['MiscFeature'].fillna('No_Feature')

        qual_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
        qual_map_BsmtFinType = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1}

        df['LotShape'] = df['LotShape'].map({'Reg': 0, 'IR1': 1, 'IR2': 2, 'IR3': 3})
        df['Condition_Artery'] = ((df['Condition1'] == 'Artery') | (df['Condition2'] == 'Artery')).astype(int)
        df['Condition_RR'] = (df['Condition1'].isin(['RRNn', 'RRAn', 'RRNe', 'RRAe']) | df['Condition2'].isin(['RRNn', 'RRAn', 'RRNe', 'RRAe'])).astype(int)
        df['Condition_Pos'] = (df['Condition1'].isin(['PosA', 'PosN']) | df['Condition2'].isin(['PosA', 'PosN'])).astype(int)
        df['ExterQual'] = df['ExterQual'].map(qual_map)
        df['ExterCond'] = df['ExterCond'].map(qual_map)
        df['BsmtQual'] = df['BsmtQual'].map({**qual_map, 'No_Basement': 0})
        df['BsmtCond'] = df['BsmtCond'].map({**qual_map, 'No_Basement': 0})
        df['BsmtExposure'] = df['BsmtExposure'].map({'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'No_Basement': 0})
        df['BsmtFinType1'] = df['BsmtFinType1'].map({**qual_map_BsmtFinType, 'No_Basement': 0})
        df['BsmtFinType2'] = df['BsmtFinType2'].map({**qual_map_BsmtFinType, 'No_Basement': 0})
        df['HeatingQC'] = df['HeatingQC'].map(qual_map)
        df['KitchenQual'] = df['KitchenQual'].map(qual_map)
        df['Functional'] = df['Functional'].map({'Typ': 7, 'Min1': 6, 'Min2': 5, 'Mod': 4, 'Maj1': 3, 'Maj2': 2, 'Sev': 1, 'Sal': 0})
        df['FireplaceQu'] = df['FireplaceQu'].map({**qual_map, 'No_Fireplace': 0})
        df['GarageFinish'] = df['GarageFinish'].map({'Fin': 3, 'RFn': 2, 'Unf': 1, 'No_Garage': 0})
        df['GarageQual'] = df['GarageQual'].map({**qual_map, 'No_Garage': 0})
        df['GarageCond'] = df['GarageCond'].map({**qual_map, 'No_Garage': 0})
        df['PoolQC'] = df['PoolQC'].map({**qual_map, 'No_Pool': 0})
        df['Fence'] = df['Fence'].map({'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, 'No_Fence': 0})
        df['MiscFeature_Shed'] = (df['MiscFeature'] == 'Shed').astype(int)
        df['MiscFeature_other'] = ((df['MiscFeature'] != 'Shed') & (df['MiscFeature'] != 'No_Feature')).astype(int)
        df['MoSold_sin'] = np.sin(2 * np.pi * df['MoSold'] / 12)
        df['MoSold_cos'] = np.cos(2 * np.pi * df['MoSold'] / 12)
        df['SaleType'] = df['SaleType'].apply(lambda x: x if x in ['WD', 'New', 'COD'] else 'Other')
        

        if self.config.general.is_train:
            drop_columns = [
                'Street', 'Utilities', 'Condition1', 'Condition2', 'RoofMatl', 'Heating',
                'LowQualFinSF', '3SsnPorch', 'PoolArea', 'MiscFeature'
            ]
            self.dataset_checkpoint['dropcols'] = drop_columns
        else:
            drop_columns = self.dataset_checkpoint['dropcols']

        df = df.drop(drop_columns, axis=1)
        self.dataset_checkpoint['dtypes'] = dict(df.dtypes)
        
        self.logger.info(f'Shape after build dataset: {df.shape}')
        self.logger.info(f'There are {df.isna().sum().sum()} missing values.')
        self.logger.info(f'There are {(df.nunique() == 2).sum()} binary columns.')

        return df
