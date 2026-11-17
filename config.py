import os
from omegaconf import OmegaConf


config = {
    'general': {
        'experiment_name': 'new_experiment',
        'seed': 42,
        'is_train': True,
        'is_rebuild_dataset': True,
        'is_save_models': True,
        'is_hold_test': True,
        'is_eval_test': True,
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
            # kwargs: estimator constructor parameters; search overrides selected values.
            # fit_kwargs: arguments passed to fit(), e.g. CatBoost cat_features.
            # val_size: internal eval_set fraction for boosting fold training, not search.
            # param_distributions: distribution specs parsed by ModelManager._make_distributions.
            'BaseLineLinearRegression': 
            {
                'kwargs': {}
            },
            'LinearRegression':
            {
                'winsorization_value': 425000,
                'kwargs': {}
            },
            'Ridge':
            {
                'winsorization_value': 425000,
                'is_hyperparameter_search': False,
                'hyperparameter_search_kwargs': {
                    'cv': 5,
                    'n_iter': 10,
                    'scoring': 'neg_root_mean_squared_log_error',
                    'param_distributions': {'alpha': {'type': 'loguniform', 'a': 0.0005, 'b': 10}}
                },
                'kwargs': {'alpha': 6, 'random_state': '${general.seed}'}
            },
            'Lasso':
            {
                'is_scale_target': True,
                'winsorization_value': 425000,
                'is_hyperparameter_search': False,
                'hyperparameter_search_kwargs': {
                    'cv': 5,
                    'n_iter': 10,
                    'scoring': 'neg_root_mean_squared_log_error',
                    'param_distributions': {'alpha': {'type': 'loguniform', 'a': 0.0005, 'b': 10}}
                },
                'kwargs': {'max_iter': 10000, 'alpha': 0.0009, 'random_state': '${general.seed}'}
            },
            'ElasticNet':
            {
                'is_scale_target': True,
                'winsorization_value': 425000,
                'is_hyperparameter_search': False,
                'hyperparameter_search_kwargs': {
                    'cv': 5,
                    'n_iter': 20,
                    'scoring': 'neg_root_mean_squared_log_error',
                    'param_distributions': {
                        'alpha': {'type': 'loguniform', 'a': 0.0005, 'b': 10},
                        'l1_ratio': {'type': 'uniform', 'a': 0.05, 'b': 0.95}
                    }
                },
                'kwargs': {
                    'max_iter': 10000,
                    'alpha': 0.0006,
                    'l1_ratio': 0.9,
                    'random_state': '${general.seed}'
                }
            },
            'CatBoostRegressor':
            {
                'val_size': 0.0,
                'fit_kwargs': {'cat_features': [
                    'Alley', 'BldgType', 'CentralAir', 'Electrical', 'Exterior1st', 'Exterior2nd',
                    'Foundation', 'GarageType', 'HouseStyle', 'LandContour', 'LandSlope',
                    'LotConfig', 'MasVnrType', 'MSSubClass', 'MSZoning', 'Neighborhood',
                    'PavedDrive', 'RoofStyle', 'SaleCondition', 'SaleType'
                ]},
                'is_hyperparameter_search': False,
                'hyperparameter_search_kwargs': {
                    'cv': 5,
                    'n_iter': 5,
                    'scoring': 'neg_root_mean_squared_log_error',
                    'param_distributions': {
                        'learning_rate': {'type': 'loguniform', 'a': 0.01, 'b': 0.05},
                        'depth': {'type': 'int', 'a': 5, 'b': 7},
                        'l2_leaf_reg': {'type': 'loguniform', 'a': 0.1, 'b': 10},
                        'random_strength': {'type': 'choice', 'values': [0.5, 1, 3, 5, 7]},
                        'bagging_temperature': {'type': 'choice', 'values': [0, 0.25, 0.5, 1, 3]}
                    }
                },
                'kwargs': {
                    'iterations': 1000,
                    'learning_rate': 0.04,
                    'depth': 6,
                    'l2_leaf_reg': 0.6,
                    'random_strength': 3,
                    'bagging_temperature': 0,
                    # 'early_stopping_rounds': 100,
                    # 'eval_metric': 'RMSE',
                    'verbose': 0,
                    'allow_writing_files': False,
                    'random_state': '${general.seed}'
                }
            },
            'LGBMRegressor':
            {
                'val_size': 0.0,
                'is_hyperparameter_search': False,
                'hyperparameter_search_kwargs': {
                    'cv': 5,
                    'n_iter': 25,
                    'scoring': 'neg_root_mean_squared_log_error',
                    'param_distributions': {
                        'learning_rate': {'type': 'loguniform', 'a': 0.01, 'b': 0.1},
                        'num_leaves': {'type': 'int', 'a': 5, 'b': 20},
                        'min_child_samples': {'type': 'int', 'a': 10, 'b': 50},
                        'subsample': {'type': 'uniform', 'a': 0.7, 'b': 1.0},
                        'colsample_bytree': {'type': 'uniform', 'a': 0.7, 'b': 1.0},
                        'reg_alpha': {'type': 'loguniform', 'a': 0.001, 'b': 10},
                        'reg_lambda': {'type': 'loguniform', 'a': 0.001, 'b': 10}
                    }
                },
                'kwargs': {
                    'n_estimators': 800,
                    'learning_rate': 0.03,
                    'num_leaves': 9,
                    'min_child_samples': 12,
                    'subsample': 0.7,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0.25,
                    'reg_lambda': 0.005,
                    'subsample_freq': 1,
                    # 'early_stopping_round': 100,
                    # 'metric': 'rmse',
                    'verbosity': -1,
                    'random_state': '${general.seed}'
                }
            },
            'XGBRegressor':
            {
                'val_size': 0.0,
                'fit_kwargs': {'verbose': 0},
                'is_hyperparameter_search': False,
                'hyperparameter_search_kwargs': {
                    'cv': 5,
                    'n_iter': 10,
                    'scoring': 'neg_root_mean_squared_log_error',
                    'param_distributions': {
                        'learning_rate': {'type': 'loguniform', 'a': 0.01, 'b': 0.1},
                        'max_depth': {'type': 'int', 'a': 3, 'b': 7},
                        'min_child_weight': {'type': 'loguniform', 'a': 1, 'b': 30},
                        'subsample': {'type': 'uniform', 'a': 0.6, 'b': 1.0},
                        'colsample_bytree': {'type': 'uniform', 'a': 0.6, 'b': 1.0},
                        'reg_alpha': {'type': 'loguniform', 'a': 0.001, 'b': 10},
                        'reg_lambda': {'type': 'loguniform', 'a': 0.001, 'b': 10},
                    }
                },
                'kwargs': {
                    'n_estimators': 1500,
                    'learning_rate': 0.036584,
                    'max_depth': 4,
                    'min_child_weight': 2.818,
                    'subsample': 0.77606,
                    'colsample_bytree': 0.97688,
                    'reg_alpha': 0.0024586,
                    'reg_lambda': 0.54567,
                    # 'early_stopping_rounds': 100,
                    # 'eval_metric': 'rmse',
                    'random_state': '${general.seed}'
                }
            },
            'DLM': 
            {
                'is_scale_target': True,
                'winsorization_value': 425000,
                'is_hyperparameter_search': False,
                'hyperparameter_search_kwargs': {
                    'cv': 5,
                    'n_iter': 10,
                    'scoring': 'neg_root_mean_squared_log_error',
                    'param_distributions': {
                        'dropout_values': {
                            'type': 'choice',
                            'values': [
                                (0.15, 0.25, 0.0),
                                (0.10, 0.25, 0.0),
                                (0.20, 0.25, 0.0),
                                (0.15, 0.20, 0.0),
                                (0.15, 0.30, 0.0),
                                (0.15, 0.25, 0.05),
                                (0.15, 0.25, 0.10)
                            ]},
                        'learning_rate': {'type': 'loguniform', 'a': 0.0001, 'b': 0.001},
                    }
                },
                'kwargs': {
                    'layer_sizes': (64, 32),
                    'dropout_values': (0.2, 0.25, 0.0),
                    'learning_rate': 0.0001,
                    'epochs': 300,
                    'plateau_epochs': 30,
                    'early_stopping': False,
                    'verbose': 100,
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
            'LinearRegression': 0,
            'Ridge': 0,
            'Lasso': 0,
            'ElasticNet': 1,
            'CatBoostRegressor': 1,
            'LGBMRegressor': 0,
            'XGBRegressor': 0.5,
            'DLM': 0.3
        },
        'stacking': {
            'winsorization_value': 500000
        },
        'to_stacking': [
            'ElasticNet',
            'CatBoostRegressor',
            'XGBRegressor',
            'DLM'
        ],
    }
}

config = OmegaConf.create(config)
