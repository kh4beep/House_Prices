import os
from omegaconf import OmegaConf


config = {
    'general': {
        'experiment_name': 'feature_engineering',
        'seed': 42,
        'is_train': True,
        'is_rebuild_dataset': True,
        'is_refit_datatransformer': True, # False reuses saved transformers; incompatible with target-derived features (MyTargetEncoder): train rows would lose cross-fitting
        'is_save_models': True,
        'is_hold_test': True,
        'is_eval_test': False,
        'test_size': 0.2,
        'n_splits': 5,
        'is_shuffle': True,
        'is_shap': False,
        'index_col': 'Id'
    },
    'paths': {
        'dataset': '../../Datasets/HousePrices/train.csv',
        'test_dataset': '../../Datasets/HousePrices/test.csv',
        'checkpoints': './checkpoints/${general.experiment_name}',
        'logs': os.path.join('${paths.checkpoints}', 'logs.log'),
        'modified_dataset': os.path.join('${paths.checkpoints}', 'dataset.csv'),
        'dataset_checkpoint': os.path.join('${paths.checkpoints}', 'dataset_checkpoint.pickle'),
        'manager_checkpoint': os.path.join('${paths.checkpoints}', 'manager_checkpoint.pickle'),
        'models': os.path.join('${paths.checkpoints}', 'models'),
        'submission': os.path.join('${paths.checkpoints}', 'submission_${general.experiment_name}.csv'),
        'oof': os.path.join('${paths.checkpoints}', 'oof.csv'),
        'importance': os.path.join('${paths.checkpoints}', 'feature_importance.csv'),
        'shap_importance': os.path.join('${paths.checkpoints}', 'shap_importance.csv')
    },
    'models': {
        'params': {
            # Per-model config (key = sklearn class name or custom alias):
            #   is_hyperparameter_search: bool
            #   hyperparameter_search_kwargs:
            #       cv, n_iter, scoring
            #       param_distributions: {param: {'type': ..., ...}}  see ModelTrainer._make_distributions
            #   fit_kwargs: dict                  passed to fit(), not to the constructor (e.g. CatBoost cat_features)
            #   val_size: float                   CatBoost only: if > 0, this fraction of train is held out as eval_set
            #   kwargs: dict                      constructor params; also the prior for hyperparameter search,
            #                                     so every searched param must be set here explicitly
            'BaseLineLinearRegression': 
            {
                'kwargs': {}
            },
            'LinearRegression':
            {
                'is_scale_target': True,
                'winsorization_value': 425000,
                'kwargs': {}
            },
            'Ridge':
            {
                'is_scale_target': True,
                'winsorization_value': 425000,
                'kwargs': {'random_state': '${general.seed}'}
            },
            'Lasso':
            {
                'is_scale_target': True,
                'winsorization_value': 425000,
                'kwargs': {'alpha': 0.0005, 'random_state': '${general.seed}'}
            },
            'ElasticNet':
            {
                'is_scale_target': True,
                'winsorization_value': 425000,
                'kwargs': {'alpha': 0.0005, 'random_state': '${general.seed}'}
            },
            'CatBoostRegressor':
            {
                'val_size': 0.0,
                'is_scale_target': True,
                'winsorization_value': 425000,
                'fit_kwargs': {'cat_features': [
                    'MSSubClass', 'MSZoning', 'Alley', 'LandContour', 'LotConfig', 'LandSlope',
                    'Neighborhood', 'BldgType', 'HouseStyle', 'RoofStyle', 'Exterior1st',
                    'Exterior2nd', 'MasVnrType', 'Foundation', 'CentralAir', 'Electrical',
                    'GarageType', 'PavedDrive', 'SaleType', 'SaleCondition'
                ]},
                'kwargs': {
                    'iterations': 100,
                    # 'early_stopping_rounds': 50,
                    # 'use_best_model': True,
                    # 'eval_metric': 'RMSE',
                    'verbose': 100,
                    'allow_writing_files': False,
                    'random_state': '${general.seed}'
                }
            },
            'LGBMRegressor':
            {
                'val_size': 0.0,
                'is_scale_target': True,
                'winsorization_value': 425000,
                'kwargs': {
                    # 'n_estimators': 2000,
                    # 'early_stopping_round': 50,
                    # 'metric': 'rmse',
                    'random_state': '${general.seed}'
                }
            },
            'XGBRegressor':
            {
                'val_size': 0.0,
                'is_scale_target': True,
                'winsorization_value': 425000,
                'fit_kwargs': {'verbose': 100},
                'kwargs': {
                    # 'n_estimators': 2000,
                    # 'early_stopping_rounds': 50,
                    # 'eval_metric': 'rmse',
                    'enable_categorical': True,
                    'tree_method': 'hist',
                    'random_state': '${general.seed}'
                }
            },
            'DLM': 
            {
                'is_scale_target': True,
                'winsorization_value': 425000,
                'kwargs': {
                    'layer_sizes': (64, 32),
                    'dropout_values': (0.15, 0.25, 0.0),
                    'learning_rate': 0.0003,
                    'epochs': 300,
                    'plateau_epochs': 30,
                    'early_stopping': False,
                    'verbose': 50,
                    'random_state': '${general.seed}'
                }
            }
        },
        'to_train': {
            'BaseLineLinearRegression': True,
            'LinearRegression': True,
            'Ridge': True,
            'Lasso': True,
            'ElasticNet': True,
            'CatBoostRegressor': True,
            'LGBMRegressor': True,
            'XGBRegressor': True,
            'DLM': True,
        },
        'to_inference': {
            'BaseLineLinearRegression': 0,
            'LinearRegression': 1,
            'Ridge': 1,
            'Lasso': 1,
            'ElasticNet': 1,
            'CatBoostRegressor': 1,
            'LGBMRegressor': 1,
            'XGBRegressor': 1,
            'DLM': 1
        },
    }
}

config = OmegaConf.create(config)
