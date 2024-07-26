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