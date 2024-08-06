import os
import pandas as pd
import numpy as np
import random
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from tqdm import tqdm

class Preprocessor:

    def __init__(self):
        pass

    def calculate_average_energy_consumption(folder_path, season_months_dict=None, type='daily'):
        """
        Process multiple parquet files in a folder, calculate average energy consumption,
        and return a pandas DataFrame with each row corresponding to one file in the folder.

        Parameters:
        - folder_path (str): Path to the folder containing parquet files.
        - season_months_dict (dict): A dictionary where keys are season names (strings) and values are lists
        of corresponding month numbers. For example, {'cold': [1, 2, 12], 'hot': [6, 7, 8], 'mild': [3, 4, 5, 9, 10, 11]}.

        Returns:
        - df_ave (pd.DataFrame): A pandas DataFrame with each row corresponding to one file in the folder (i.e. one building).
        The columns are multi-layer with the first layer being the day/week/month/season and the second layer the hour of the day 
        Index ('bldg_id') contains building IDs. Column values are average hourly electricity energy consumption
        """
        # Initialize an empty list to store individual DataFrames for each file
        result_dfs = []

        # Iterate through all files in the folder_path
        for file_name in tqdm(os.listdir(folder_path)):
            if file_name.endswith(".parquet"):
                # Extract the bldg_id from the file name
                bldg_id = int(file_name.split('.')[0])

                # Construct the full file path
                file_path = os.path.join(folder_path, file_name)

                # Read the original parquet file
                df = pd.read_parquet(file_path)

                # Convert 'timestamp' column to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['hour'] = df['timestamp'].dt.hour

                if type == 'daily':# -> goes from Input: 365 * 24 * 4 = 35,040 columns to 365 * 24 = 8,760 values per building
                    df['day_of_year'] = df['timestamp'].dt.day_of_year
                    df['hourly_energy_consumption'] = df.groupby(['day_of_year', 'hour'])['out.electricity.total.energy_consumption'].transform('mean')
                    result_df = df.pivot_table(values='hourly_energy_consumption', index='bldg_id', columns=['day_of_year', 'hour'])
                
                elif type == 'weekly':# -> goes from Input: 365 * 24 * 4 = 35,040 columns to 52 * 24 = 1,248 values per building
                    df['week'] = df['timestamp'].dt.isocalendar().week
                    df['weekly_energy_consumption'] = df.groupby(['week', 'hour'])['out.electricity.total.energy_consumption'].transform('mean')
                    result_df = df.pivot_table(values='weekly_energy_consumption', index='bldg_id', columns=['week', 'hour'])

                elif type == 'monthly':# -> goes from Input: 365 * 24 * 4 = 35,040 columns to 12 * 24 = 288 values per building
                    df['month'] = df['timestamp'].dt.month
                    df['monthly_energy_consumption'] = df.groupby(['month', 'hour'])['out.electricity.total.energy_consumption'].transform('mean')
                    result_df = df.pivot_table(values='monthly_energy_consumption', index='bldg_id', columns=['month', 'hour'])

                elif type == 'seasonal': # originally provided prerpocessing method -> goes from Input: 365 * 24 * 4 = 35,040 columns to 365 * (12/s)  = ... values per building
                    df['month'] = df['timestamp'].dt.month
                    # Create a mapping from month to the corresponding season
                    month_to_season = {month: season for season, months_list in season_months_dict.items() for month in months_list}

                    # Assign a season to each row based on the month
                    df['season'] = df['month'].map(month_to_season)

                    # Calculate hourly average energy consumption for each row
                    df['hourly_avg_energy_consumption'] = 4 * df.groupby(['season', 'hour'])['out.electricity.total.energy_consumption'].transform('mean')

                    # Pivot the dataframe to create the desired output format
                    result_df = df.pivot_table(values='hourly_avg_energy_consumption', index='bldg_id', columns=['season', 'hour'])

                    # Reset the column names
                    result_df.columns = pd.MultiIndex.from_tuples([(season, hour+1) for season, months_list in season_months_dict.items() for hour in range(24)])
                else:
                    raise ValueError('Invalid type. Please select from hourly, weekly, or monthly.')

                # Add 'bldg_id' index with values corresponding to the names of the parquet files
                result_df['bldg_id'] = bldg_id
                result_df.set_index('bldg_id', inplace=True)

                # Append the result_df to the list
                result_dfs.append(result_df)

        # Concatenate all individual DataFrames into a single DataFrame
        df_ave = pd.concat(result_dfs, ignore_index=False)

        return df_ave
    
    def load_standard_df(folder_path):
        """
        Process multiple parquet files in a folder and return a pandas DataFrame with each row corresponding to one file in the folder.

        Parameters:
        - folder_path (str): Path to the folder containing parquet files.
        
        Returns:
        - df (pd.DataFrame): A pandas DataFrame with each row corresponding to one file in the folder (i.e. one building).
        """
        # Initialize an empty list to store individual DataFrames for each file
        result_dfs = []

        # Iterate through all files in the folder_path
        for file_name in tqdm(os.listdir(folder_path)):
            if file_name.endswith(".parquet"):
                # Extract the bldg_id from the file name
                bldg_id = int(file_name.split('.')[0])

                # Construct the full file path
                file_path = os.path.join(folder_path, file_name)

                # Read the original parquet file
                df = pd.read_parquet(file_path)

                # Convert 'timestamp' column to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                result_df = df.pivot_table(values='out.electricity.total.energy_consumption', index='bldg_id', columns=['timestamp'])

                # Add 'bldg_id' index with values corresponding to the names of the parquet files
                result_df['bldg_id'] = bldg_id
                result_df.set_index('bldg_id', inplace=True)

                # Append the result_df to the list
                result_dfs.append(result_df)

        # Concatenate all individual DataFrames into a single DataFrame
        output_df = pd.concat(result_dfs, ignore_index=False)

        return output_df
    
class TargetPreprocessor:

    def __init__(self):
        pass

    def preprocess_com(self, df_com):
        com_exception_cols = ["in.number_of_stories_com", "in.vintage_com", "in.tstat_clg_sp_f..f_com", "in.tstat_htg_sp_f..f_com", 
                            "in.weekday_opening_time..hr_com", "in.weekday_operating_hours..hr_com"]

        # columns that are simply transformed to numeric
        for col in com_exception_cols:
            if col == "in.vintage_com":
                period_dict = {'Before 1946': 1940,  # Using an estimated middle year
                                '1946 to 1959': 1952,
                                '1960 to 1969': 1965,
                                '1970 to 1979': 1975,
                                '1980 to 1989': 1985,
                                '1990 to 1999': 1995,
                                '2000 to 2012': 2006,
                                '2013 to 2018': 2016
                                }
                df_com['in.vintage_com'] = df_com['in.vintage_com'].map(period_dict)
            else:
                df_com[col] = pd.to_numeric(df_com[col])

        # One-hot encode categorical features
        one_hot_encoder = OneHotEncoder(sparse_output=False)
        categorical_encoded = one_hot_encoder.fit_transform(df_com[df_com.columns.difference(com_exception_cols)])

        association_dict = {}
        column_map = list(one_hot_encoder.get_feature_names_out(df_com.columns.difference(com_exception_cols)))
        for column in column_map:
            # Extract the base attribute name from the column name
            # Assuming the format 'prefix_attribute_com_something'
            base_attribute = column.split('_com_')[0]  # Split by '_com_' to get the base attribute
            
            if base_attribute not in association_dict:
                association_dict[base_attribute] = []
            
            association_dict[base_attribute].append(column_map.index(column)+len(com_exception_cols))

        categorical_encoded = pd.DataFrame(categorical_encoded)
        categorical_encoded.index = df_com.index
        df_com = pd.concat([df_com[com_exception_cols], pd.DataFrame(categorical_encoded)], axis=1)

        return df_com.fillna(0), association_dict, one_hot_encoder


    def preprocess_res(self, df_res):
        res_exception_cols = ["in.bedrooms_res", "in.cooling_setpoint_res", "in.heating_setpoint_res", "in.geometry_floor_area_res", 
                            "in.income_res", "in.vintage_res"]
        
        df_res["in.bedrooms_res"] = pd.to_numeric(df_res['in.bedrooms_res'])
        
        # handling the temperature columns f.e. going from 78F to 78
        df_res['in.cooling_setpoint_res'] = df_res['in.cooling_setpoint_res'].apply(lambda temp: float(temp[:-1]))
        df_res['in.heating_setpoint_res'] = df_res['in.heating_setpoint_res'].apply(lambda temp: float(temp[:-1]))

        def convert_geometry_to_mean(area):
            if '+' in area:
                return float(area[:-1])  # Take the numeric part of '4000+'
            lower, upper = map(int, area.split('-'))
            return int((lower + upper) / 2)

        df_res['in.geometry_floor_area_res'] = df_res['in.geometry_floor_area_res'].apply(convert_geometry_to_mean)

        def convert_income_range_to_mean(income):
            if '+' in income:
                return float(income[:-1])  # Take the numeric part of '200000+'
            if '<' in income:
                return float(income[1:])  # Take the numeric part of '<10000'
            
            lower, upper = map(int, income.split('-'))
            return int((lower + upper) / 2)

        df_res['in.income_res'] = df_res['in.income_res'].apply(convert_income_range_to_mean)

        vintage_mapping = {'<1940': 1940, '1940s': 1945, '1950s': 1955, '1960s': 1965,
                           '1970s': 1975, '1980s': 1985, '1990s': 1995, '2000s': 2005, '2010s': 2015}

        df_res['in.vintage_res'] = df_res['in.vintage_res'].map(vintage_mapping)

        # One-hot encode categorical features
        one_hot_encoder = OneHotEncoder(sparse_output=False)
        categorical_encoded = one_hot_encoder.fit_transform(df_res[df_res.columns.difference(res_exception_cols)])

        association_dict = {}
        column_map = list(one_hot_encoder.get_feature_names_out(df_res.columns.difference(res_exception_cols)))
        for column in column_map:
            # Extract the base attribute name from the column name
            # Assuming the format 'prefix_attribute_res_something'
            base_attribute = column.split('_res_')[0]  # Split by '_res_' to get the base attribute
            
            if base_attribute not in association_dict:
                association_dict[base_attribute] = []
            
            association_dict[base_attribute].append(column_map.index(column)+len(res_exception_cols))

        categorical_encoded = pd.DataFrame(categorical_encoded)
        categorical_encoded.index = df_res.index
        df_res = pd.concat([df_res[res_exception_cols], pd.DataFrame(categorical_encoded)], axis=1)

        return df_res.fillna(0), association_dict, one_hot_encoder