import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix

import os
import sys
base_path = r"C:\Users\KAI\Coding\ThinkOnward_challenge\thinkOnward_TSClassification"
data_path = r"\data\building-instinct-starter-notebook\Starter notebook"
sys.path.append(base_path+data_path)
sys.path.append(base_path+"\kai")

from models import MultiTaskLSTM, CustomLoss, TimeSeriesDataset
from preprocessing.preprocessing import TargetPreprocessor
from postprocessing import inverse_process_res, inverse_process_com
from preprocessing.utils import print_gpu_memory, free_gpu_memory


def simple_classification(X, y, X_test, print_performance=False):
    """
    Simple Random Forest classifier for binary classification.
    """

    if print_performance:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = Pipeline([('preprocessor', ColumnTransformer([
                    ('scaler', StandardScaler(), X.columns),
                    ('encoder', OneHotEncoder(), [])
                ])),
                ('classifier', RandomForestClassifier(random_state=42))
            ])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        print(f1_score(y_val, y_pred, average='macro'))
        print(confusion_matrix(y_val, y_pred))

    clf = Pipeline([('preprocessor', ColumnTransformer([
                ('scaler', StandardScaler(), X.columns),
                ('encoder', OneHotEncoder(), [])
            ])),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
    clf.fit(X, y)
    y_pred = clf.predict(X_test)
    return y_pred

def train_lstm(X, y, parameters):
    """
    Train a LSTM model on the given data. Either for residential or commercial buildings.
    """

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.cuda.empty_cache()

    # disentangle the parameters dict
    batch_size = parameters["batch_size"]
    hidden_size = parameters["hidden_size"]
    num_epochs = parameters["num_epochs"]
    weight_numerical = parameters["weight_numerical"]
    weight_categorical = parameters["weight_categorical"]
    association_dict = parameters["association_dict"]
    unique_values = parameters["unique_values"]
    device = parameters["device"]
    lr = parameters["lr"]
    
    if device != 'cpu':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    input_size, sequence_length = 1, X.shape[1]
    num_classes_categorical = y.shape[1]

    best_combined_loss = float('inf')
    building_type = list(unique_values.keys())[0][-3:] # either 'res' or 'com'
    checkpoint_filename = base_path+ f'/kai/checkpoints/{building_type}_model_checkpoint_{datetime.now().strftime("%m_%d_%H_%M")}.pth.tar'

    # create dataloaders
    dataloader = DataLoader(TimeSeriesDataset(X, y), batch_size=batch_size, shuffle=True)

    model = MultiTaskLSTM(input_size, hidden_size, num_classes_categorical)
    model = model.to(device)
    criterion = CustomLoss(association_dict, unique_values)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_categorical_batch in dataloader:
            X_batch = X_batch.view(X_batch.shape[0], sequence_length, input_size)
            X_batch = X_batch.to(device)
            y_categorical_batch = y_categorical_batch.to(device)
            optimizer.zero_grad()
            categorical_pred = model(X_batch)
            loss, loss_numerical, loss_categorical = criterion(categorical_pred, y_categorical_batch)
            
            # Combine losses with adjusted weights
            combined_loss = weight_numerical * loss_numerical + weight_categorical * loss_categorical

            combined_loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], '
                    f'Total Loss: {loss.item():.4f}, '
                    f'Numerical Loss: {loss_numerical.item():.4f}, '
                    f'Categorical Loss: {loss_categorical.item():.4f}, '
                    f'Combined Loss: {combined_loss.item():.4f}, '
                    f'weights: {weight_numerical:.4f}, {weight_categorical:.4f}')
        
        if combined_loss.item() < best_combined_loss:
            torch.save({'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': combined_loss.item(),}, checkpoint_filename)
    return model, checkpoint_filename

def predict_lstm(df_test, parameters, encoder, filename='/kai/checkpoints/com_model_checkpoint_08_13_23_14.pth.tar'):
    free_gpu_memory()
    X_test = torch.tensor(df_test[df_test.columns.difference(["building_stock_type"])].values, dtype=torch.float32)
    # disentangle the parameters dict
    batch_size = parameters["batch_size"]
    hidden_size = parameters["hidden_size"]
    device = parameters["device"]
    association_dict = parameters["association_dict"]
    num_classes_categorical = parameters["num_classes_categorical"]
    input_size, sequence_length = 1, X_test.shape[1]
    unique_values = parameters["unique_values"]
    building_type = list(unique_values.keys())[0][-3:] # either 'res' or 'com'

    model = MultiTaskLSTM(input_size, hidden_size, num_classes_categorical)
    model = model.to(device)
    checkpoint_filename = filename
    checkpoint = torch.load(checkpoint_filename)
    model.load_state_dict(checkpoint['model_state_dict'])

    predictions = []
    dataloader = DataLoader(TimeSeriesDataset(X_test, torch.zeros(X_test.shape[0], num_classes_categorical)), batch_size=batch_size, shuffle=False)

    print_gpu_memory()
    for X_batch, _ in tqdm(dataloader):
        model.eval()
        _X_batch = X_batch.view(X_batch.shape[0], sequence_length, input_size)
        _X_batch = _X_batch.to(device)
        with torch.no_grad():
            categorical_pred = model.predict(_X_batch, association_dict)
        predictions.append(categorical_pred.cpu())
        # del _X_batch, categorical_pred
        # free_gpu_memory()
    predictions = torch.cat(predictions, dim=0)
    if building_type == 'res':
        arr_df = inverse_process_res(predictions, encoder)
    else:
        arr_df = inverse_process_com(predictions, encoder)
    arr_df.index = df_test.index
    return arr_df

def create_submission(df_com, df_res, df_test, save_filepath=None):
    """
    Given a df_test dataframe that already contains the predictions for the building_stock_type column,
    and two dataframes df_com and df_res that contain the predictions for the residential and commercial
    columns respectively, this function will create a submission dataframe that is compatible with the submission format.
    """
    # First load the training labels again to get the correct column order
    load_filepath_labels = os.path.join(base_path + data_path,'building-instinct-train-label', 'train_label.parquet')
    df_targets = pd.read_parquet(load_filepath_labels, engine='pyarrow')
    df_targets.sort_index(inplace=True)

    # Create a new dataframe with the same index as df_targets
    bldg_id_list = [i for i in range(1,1441)]
    df = pd.DataFrame(index=bldg_id_list, columns=df_targets.columns)
    df.index.name = df_targets.index.name

    # Populate the first column 'building_stock_type'
    df['building_stock_type'] = df_test["building_stock_type"].map({0: 'residential', 1: 'commercial'})

    res_columns = [col for col in df_targets.columns if col.endswith('_res')]
    com_columns = [col for col in df_targets.columns if col.endswith('_com')]
    for bldg_id in df.index:
        if df.at[bldg_id, 'building_stock_type'] == 'residential':
            df.loc[bldg_id, com_columns] = np.nan
            for col in res_columns:
                df.at[bldg_id, col] = df_res.at[bldg_id, col]
        else:
            df.loc[bldg_id, res_columns] = np.nan
            for col in com_columns:
                df.at[bldg_id, col] = df_com.at[bldg_id, col]
    df = df.astype(str)
    if save_filepath:
        df.to_parquet(save_filepath)
    return df

def unique_values(df):
    unique_values = {}
    for col in df.columns:
        if df[col].nunique() > 2:
            unique_values[col] = df[col].unique()
    return unique_values

def main():
    base_path = r"C:\Users\KAI\Coding\ThinkOnward_challenge\thinkOnward_TSClassification"
    data_path = r"\data\building-instinct-starter-notebook\Starter notebook"
    sys.path.append(base_path+data_path)
    sys.path.append(base_path+"\kai")
    df_features = pd.read_parquet(base_path + '/preprocessed_data/standard_data.parquet', engine='pyarrow')
    df_features.sort_index(inplace=True)

    load_filepath_labels = os.path.join(base_path + data_path,'building-instinct-train-label', 'train_label.parquet')#path to the train label file
    df_targets = pd.read_parquet(load_filepath_labels, engine='pyarrow')

    # RandomForest classifier
    X, y = df_features, df_targets["building_stock_type"].map({"residential": 0, "commercial": 1})
    df_test = pd.read_parquet(base_path + '/preprocessed_data/data_test.parquet', engine='pyarrow')
    df_test.sort_index(inplace=True)

    y_pred = simple_classification(X, y, df_test, print_performance=True)
    df_test["building_stock_type"] = y_pred

    # The 2 LSTM models
    # 1. Preprocessing
    df_targets_res = df_targets[df_targets.building_stock_type == "residential"].filter(like='_res').copy()
    df_targets_com = df_targets[df_targets.building_stock_type == "commercial"].filter(like='_com').copy()
    target_preprocessor = TargetPreprocessor()
    df_targets_res, association_dict_res, encoder_res = target_preprocessor.preprocess_res(df_targets_res)
    df_targets_com, association_dict_com, encoder_com = target_preprocessor.preprocess_com(df_targets_com)
    unique_values_res = unique_values(df_targets_res)
    unique_values_com = unique_values(df_targets_com)

    common_indices = df_features.index.intersection(df_targets_com.index)
    X_com = df_features[df_features.index.isin(common_indices)]
    X_com = torch.tensor(X_com.values, dtype=torch.float32)
    y_com = df_targets_com[df_targets_com.index.isin(common_indices)]
    y_com = torch.tensor(y_com.values, dtype=torch.float32)

    X_res = df_features[df_features.index.isin(df_targets_res.index)]
    X_res = torch.tensor(X_res.values, dtype=torch.float32)
    y_res = df_targets_res[df_targets_res.index.isin(df_features.index)]
    y_res = torch.tensor(y_res.values, dtype=torch.float32)

    # 2. Training
    parameters = {
        "batch_size" : 16,
        "hidden_size" : 82,
        "num_epochs" : 25,
        "weight_numerical" : 10e-4,
        "weight_categorical" : 1.0,
        "association_dict": association_dict_com,
        "unique_values": unique_values_com,
        "device": "cuda",
        "lr": 0.1,
        "num_classes_categorial":y_com.shape[1],}
    model_com, filename_com = train_lstm(X_com, y_com, parameters)

    parameters = {
        "batch_size" : 16,
        "hidden_size" : 82,
        "num_epochs" : 25,
        "weight_numerical" : 10e-5,
        "weight_categorical" : 1.0,
        "association_dict": association_dict_res,
        "unique_values": unique_values_res,
        "device": "cuda",
        "lr": 0.1,
        "num_classes_categorial":y_res.shape[1],}
    model_res, filename_res = train_lstm(X_res, y_res, parameters)

    # 3. Predictions
    df_test_com = df_test[df_test.building_stock_type == 1]
    X_test_com, _ = torch.tensor(df_test_com[df_test_com.columns.difference(["building_stock_type"])].values, dtype=torch.float32), df_test_com["building_stock_type"]
    arr_com = predict_lstm(X_test_com, parameters, encoder_com, filename=filename_com)

    df_test_res = df_test[df_test.building_stock_type == 0]
    X_test_res, _ = torch.tensor(df_test_res[df_test_res.columns.difference(["building_stock_type"])].values, dtype=torch.float32), df_test_res["building_stock_type"]
    arr_res = predict_lstm(X_test_res, parameters, encoder_res, filename=filename_res)
    # 4. Submission
    df = create_submission(arr_com, arr_res, df_test, save_filepath="submission.parquet")