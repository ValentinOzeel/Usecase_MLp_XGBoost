## Define the XGB model
from xgboost import XGBClassifier
# Import MLp's config function
from MLp.conf.config_functions import get_config
mlp_config = get_config()
RANDOM_STATE = mlp_config['MLP_CONFIG']['RANDOM_STATE']


''' Model with already finetuned hyperparams (no need to use finetuned hyperparams here, you can just run optuna with MLp)'''
xgb_model = XGBClassifier(
                random_state=RANDOM_STATE,
                n_estimators=455, 
                learning_rate=0.05997191158460472, 
                max_depth=5, 
                subsample=0.7800258644252909, 
                colsample_bytree=0.7495457287282703, 
                min_child_weight=1, 
                reg_alpha=0.0039758160070881374, 
                reg_lambda=0.005927768559251354, 
                gamma=0.004110407481078195
)