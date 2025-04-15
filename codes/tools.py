import os
import torch
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
from transformers import AutoTokenizer, AutoModel


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def txt_level_concat(single_or_merge):
    concat_template1 = """
    话题标题：{}
    大模型对关于该话题{}的热度预测指示：{}
    """
    concat_template2 = """
    话题标题：{}
    发布时间：{}
    大模型对关于该话题{}的热度预测指示：{}
    """
    concat_template3 = """
    话题标题：{}
    发布时间：{}
    大模型对关于该话题类型、目标群众、发布时间三种属性的分析以及综合考虑后对该话题热度预测的指示：{}
    """
    df = pd.read_csv('../data/o3_instruction.csv')
    if single_or_merge:     # single attribute
        merge_list = ['merge_cat', 'merge_aud', 'merge_time']
        attr_name_list = ['category_instruction', 'audience_instruction', 'time_instruction']
        df['merge_cat'] = df.apply(lambda row: concat_template1.format(row['title'], '类型', row['category_instruction']), axis=1)
        df['merge_aud'] = df.apply(lambda row: concat_template1.format(row['title'], '目标群众', row['audience_instruction']), axis=1)
        df['merge_time'] = df.apply(lambda row: concat_template2.format(row['title'], row['datetime'], '发布时间', row['time_instruction']), axis=1)
    else:   # merged attribute
        merge_list = ['merge_all']
        attr_name_list = ['merge_instruction']
        df['merge_all'] = df.apply(lambda row: concat_template3.format(row['title'], row['datetime'], row['merge_instruction']), axis=1)
    return merge_list, attr_name_list, df


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def get_embedding_output(tokenizer, embedding_layer, xtrain, xtest):
    train_embedding_list, test_embedding_list = [], []
    for a in xtrain:
        inputs = tokenizer(a, return_tensors="pt", truncation=True, max_length=128)
        input_ids = inputs["input_ids"]

        # get embedding layer output
        with torch.no_grad():
            embedding_layer_output = embedding_layer(input_ids)  
            embedding_layer_output = embedding_layer_output.to(torch.float).cpu()
        train_embedding_list.append(embedding_layer_output.mean(dim=1).squeeze().numpy())  # average pooling

    for b in xtest:
        inputs = tokenizer(b, return_tensors="pt", truncation=True, max_length=128)
        input_ids = inputs["input_ids"]

        # get embedding layer output
        with torch.no_grad():
            embedding_layer_output = embedding_layer(input_ids)  
            embedding_layer_output = embedding_layer_output.to(torch.float).cpu()
        test_embedding_list.append(embedding_layer_output.mean(dim=1).squeeze().numpy())  # average pooling
    
    return train_embedding_list, test_embedding_list


def getXnY(train_data, test_data, merge, tokenizer, embedding_layer):
    X_train_texts = train_data[merge]
    y_train = train_data['browse_log_norm']
    X_test_texts = test_data[merge]
    y_test = test_data['browse_log_norm']
    train_embedding_list, test_embedding_list = get_embedding_output(tokenizer, embedding_layer, X_train_texts, X_test_texts)

    # numpy array (batch_size, num_hiddens)
    X_train, X_test = np.array(train_embedding_list), np.array(test_embedding_list)
    print(f"feature matrix shape: {X_train.shape}")
    return X_train, X_test, y_train, y_test


def train(single_or_merge, param_list, model_list, name_list, LLM):
    merge_list, attr_name_list, df = txt_level_concat(single_or_merge)
    split_index = int(len(df) * 0.8)
    train_data = df[:split_index]  # train set
    test_data = df[split_index:]   # test set
    rmse_scorer = make_scorer(rmse, greater_is_better=False)

    # Tokenization and Embedding
    tokenizer = AutoTokenizer.from_pretrained(LLM, trust_remote_code=True)
    LLM_model = AutoModel.from_pretrained(LLM, trust_remote_code=True)
    embedding_layer = LLM_model.get_input_embeddings()  # get embedding layer

    for merge, attr in zip(merge_list, attr_name_list):
        X_train, X_test, y_train, y_test = getXnY(train_data, test_data, merge, tokenizer, embedding_layer)

        # scalar
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        for ML_model, param, name in zip(model_list, param_list, name_list):
            # RandomizedSearchCV configuration
            search = RandomizedSearchCV(
                estimator=ML_model,
                param_distributions=param,
                n_iter=30,                # iter tinmes
                cv=3,                     # cross validation k
                scoring=rmse_scorer,      # RMSE
                n_jobs=-1,                # use all CPU cores
                random_state=42,          # random seed
                verbose=2                 # print log
            )
            print(f"start training {name}")
            
            search.fit(X_train, y_train)
            ML_model.fit(X_train, y_train)

            # model prediction    
            y_pred = search.predict(X_test)      
            y_pred_series = pd.Series(y_pred)

            # save the prediction results and trained ML models
            y_pred_series.to_csv(f'../browse_trained_results/o3_{attr}_{name}_txt_level.csv', index=False) # training results
            joblib.dump(search, f'../ML_trained_models/o3_{attr}_{name}.pkl')  # trained models



def evaluate_by_csv(single_or_merge, name_list):
    _, attr_name_list, df = txt_level_concat(single_or_merge)
    split_index = int(len(df) * 0.8)
    test_data = df[split_index:]   # test set
    res = []

    for attr in attr_name_list:
        y_test = test_data['browse_log_norm'].values
        for name in name_list:
            y_pred = pd.read_csv(f'../browse_trained_results/o3_{attr}_{name}_txt_level.csv')
            y_pred = y_pred.squeeze()
            mse = mean_squared_error(y_test, y_pred)    # mse
            mae = round(mean_absolute_error(y_test, y_pred), 4)   # mae

            # calculate average and standard variance value
            P_mean = np.mean(y_test)  
            P_pred_mean = np.mean(y_pred)  
            sigma_P = np.std(y_test, ddof=1)  
            sigma_P_pred = np.std(y_pred, ddof=1)  

            # standardize and calculate SRC
            k = len(y_test)  # sample volume
            src = (1 / (k - 1)) * np.sum(
                ((y_test - P_mean) / sigma_P) * ((y_pred - P_pred_mean) / sigma_P_pred)
            )
            src = round(src, 4)
            res.append(f'merge_attribute: {attr}, model: {name}, RMSE: {round(mse ** 0.5, 4)}, MAE: {mae}, SRC: {src}')

    for r in res:
        print(r)


def evaluate_by_pkl(single_or_merge, name_list, LLM):
    merge_list, attr_name_list, df = txt_level_concat(single_or_merge)
    split_index = int(len(df) * 0.8)
    train_data = df[:split_index]  # train set
    test_data = df[split_index:]   # test set
    
    res = []

    # Tokenization and Embedding
    tokenizer = AutoTokenizer.from_pretrained(LLM, trust_remote_code=True)
    LLM_model = AutoModel.from_pretrained(LLM, trust_remote_code=True)
    embedding_layer = LLM_model.get_input_embeddings()  # get embedding layer

    for merge, attr in zip(merge_list, attr_name_list):
        X_train, X_test, y_train, y_test = getXnY(train_data, test_data, merge, tokenizer, embedding_layer)

        # scalar
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        for name in name_list:
            # load trained ML model
            print(f'load model: o3_{attr}_{name}.pkl')
            search = joblib.load(f'../ML_trained_models/o3_{attr}_{name}.pkl')  
            
            # model prediction    
            y_pred = search.predict(X_test)      

            # evaluate
            mse = mean_squared_error(y_test, y_pred)    # mse
            mae = round(mean_absolute_error(y_test, y_pred), 4)   # mae

            # calculate average and standard variance value
            P_mean = np.mean(y_test)  
            P_pred_mean = np.mean(y_pred)  
            sigma_P = np.std(y_test, ddof=1)  
            sigma_P_pred = np.std(y_pred, ddof=1)  

            # standardize and calculate SRC
            k = len(y_test)  # sample volume
            src = (1 / (k - 1)) * np.sum(
                ((y_test - P_mean) / sigma_P) * ((y_pred - P_pred_mean) / sigma_P_pred)
            )
            src = round(src, 4)
            res.append(f'merge_attribute: {attr}, model: {name}, RMSE: {round(mse ** 0.5, 4)}, MAE: {mae}, SRC: {src}')


    for r in res:
        print(r)
    return 