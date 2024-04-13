### For grid_search (DICT)
## Define the parameter grid for hyperparameter tuning
model_grid = {
    ###### Preprocessor transformer's parameter tuning
    'preprocessing__kmeans_clustering_transformer-1__n_clusters': [10, 25],
    
    ###### Model's hyperparameter tuning
    'model__learning_rate': [0.2],
    'model__n_estimators': [50, 200],
#    'model__max_depth': [3],
#   'model__reg_alpha': [0.001, 0.001, 0.1, 0.5],
#   'model__reg_lambda': [0.001, 0.001, 0.1, 1, 3],
}




## For optuna (LIST)
optuna_hyperparameters = [
    ###### Model's hyperparameter tuning
    ('model__n_estimators', 'suggest_int', (50, 500), {}),
    ('model__learning_rate', 'suggest_float', (0.001, 0.3), {'log': True}),
    ('model__max_depth', 'suggest_int', (3, 12), {}),
  #  ('model__booster', 'suggest_categorical', (['gbtree', 'gblinear', 'dart']), {}),
    ('model__subsample', 'suggest_float', (0.1, 1.0), {}),
    ('model__colsample_bytree', 'suggest_float', (0.1, 1.0), {}),
    ('model__min_child_weight', 'suggest_int', (1, 10), {}),   
    ('model__reg_alpha', 'suggest_float', (0.0001, 1), {'log': True}),
    ('model__reg_lambda', 'suggest_float', (0.0001, 1), {'log': True}),
    ('model__gamma', 'suggest_float', (0.0001, 1), {'log': True}),
    ###### Preprocessor transformer's parameter tuning
    ('preprocessing__kmeans_clustering_transformer-1__n_clusters', 'suggest_int', (2, 50), {}),
 #   ('preprocessing__pca_transformer-1__n_components', 'suggest_int', (2, 9), {}),
]