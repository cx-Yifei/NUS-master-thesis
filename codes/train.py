from tools import train
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint, uniform, loguniform

# single_or_merge = 0     # train merged attributes
single_or_merge = 1     # train single attributes


# parameter list
param_dist_rf = {
    "n_estimators": randint(100, 500),
    "max_depth": [None] + list(randint(5, 20).rvs(10)),
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(5, 30),
    "max_features": uniform(0.1, 0.9),
    "bootstrap": [True, False],
    "criterion": ["squared_error"]  # RMSE
}

param_dist_lgb = {
    "n_estimators": randint(50, 300),
    "learning_rate": loguniform(1e-3, 0.1),
    "num_leaves": randint(20, 80),
    "max_depth": randint(3, 8),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
    "reg_alpha": loguniform(1e-5, 1),
    "reg_lambda": loguniform(1e-5, 1),
    "min_child_samples": randint(20, 100),
    "objective": ["regression"],                  # regression task
    "metric": ["rmse"]                            # RMSE
}

param_dist_cat = {
    "iterations": randint(100, 500),          
    "depth": randint(4, 12),                   
    "learning_rate": loguniform(1e-3, 0.1),    
    "l2_leaf_reg": loguniform(1e-3, 10),       
    "border_count": randint(32, 255),          
    "random_strength": uniform(1e-3, 10),      
    "grow_policy": ["SymmetricTree", "Depthwise", "Lossguide"]  
}
param_list = [param_dist_rf, param_dist_lgb, param_dist_cat]

# ML models
model_list = [RandomForestRegressor(random_state=42),
            LGBMRegressor(random_state=42, verbosity=-1),
            CatBoostRegressor(random_state=42)]
name_list = ['RandomForest', 'LightGBM', 'CatBoost']

# roberta tokenizer and model name
LLM = 'FacebookAI/xlm-roberta-base'

train(single_or_merge, param_list, model_list, name_list, LLM)