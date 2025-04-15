from tools import *

# single_or_merge = 0     # train merged attributes
single_or_merge = 1     # train single attributes

name_list = ['RandomForest', 'LightGBM', 'CatBoost']
LLM = 'FacebookAI/xlm-roberta-base'

evaluate_by_csv(single_or_merge, name_list)
print("\n----------------------------------------\n")
evaluate_by_pkl(single_or_merge, name_list, LLM)