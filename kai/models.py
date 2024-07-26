import os
import pandas as pd
import numpy as np
import random
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
def train_model(X, y):
    """
    Train hierarchical classification models for predicting building stock types and their respective attributes.

    This function trains three separate models:
    1. A classifier to predict the 'building_stock_type' (either 'commercial' or 'residential').
    2. A classifier for predicting attributes of commercial buildings.
    3. A classifier for predicting attributes of residential buildings.

    The function preprocesses the input data using standard scaling and optional one-hot encoding before training the classifiers.

    Parameters:
    ----------
    X : pd.DataFrame
        The feature dataframe used for training. Each row represents a building, and each column represents a feature.
    
    y : pd.DataFrame
        The target dataframe containing the labels. It includes the 'building_stock_type' column and other columns ending
        with '_com' for commercial attributes and '_res' for residential attributes.

    Returns:
    -------
    list
        A list of three trained classifiers:
        1. classifier_type: A RandomForestClassifier model for predicting 'building_stock_type'.
        2. classifier_residential: A RandomForestClassifier model for predicting residential attributes.
        3. classifier_commercial: A RandomForestClassifier model for predicting commercial attributes.

    """
    
    # Define column transformers for commercial and residential buildings
    transformer_commercial = ColumnTransformer([
        ('scaler', StandardScaler(), X.columns),
        ('encoder', OneHotEncoder(), [])
    ])
    
    transformer_residential = ColumnTransformer([
        ('scaler', StandardScaler(), X.columns),
        ('encoder', OneHotEncoder(), [])
    ])
    
    # Filter features and targets for commercial and residential buildings
    X_commercial = X[y['building_stock_type'] == 'commercial']
    X_residential = X[y['building_stock_type'] == 'residential']
    y_commercial = y[y['building_stock_type'] == 'commercial'].filter(like='_com')
    y_residential = y[y['building_stock_type'] == 'residential'].filter(like='_res')
    
    # Train classifier to predict 'building_stock_type'
    classifier_type = Pipeline([
        ('preprocessor', ColumnTransformer([
            ('scaler', StandardScaler(), X.columns),
            ('encoder', OneHotEncoder(), [])
        ])),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    classifier_type.fit(X, y['building_stock_type'])
    
    # Train separate classifiers for commercial and residential buildings
    classifier_commercial = Pipeline([
        ('preprocessor', transformer_commercial),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    classifier_residential = Pipeline([
        ('preprocessor', transformer_residential),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Train models
    classifier_commercial.fit(X_commercial, y_commercial)
    classifier_residential.fit(X_residential, y_residential)
    
    return [classifier_type, classifier_residential, classifier_commercial]