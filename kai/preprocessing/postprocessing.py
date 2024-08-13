import numpy as np
import pandas as pd

def map_to_closest_values(predictions, possible_values):
    """
    Map predicted values to the closest possible values.

    Args:
    - predictions (list of lists): The predicted values.
    - possible_values (dict): A dictionary where keys are column names and values are lists of possible values.

    Returns:
    - mapped_predictions (list of lists): The predictions mapped to the closest possible values.
    """
    def closest_value(predicted, possible):
        return possible[np.argmin((np.array(possible) - predicted)**2)]
    
    mapped_predictions = []
    
    for prediction in predictions:
        mapped_row = []
        for col_name, pred_value in zip(possible_values.keys(), prediction):
            mapped_value = closest_value(pred_value, possible_values[col_name])
            mapped_row.append(mapped_value)
        mapped_predictions.append(mapped_row)
    
    return mapped_predictions

def inverse_process(prediction, encoder):
    prediction = prediction.cpu().detach().numpy()
    # Possible values for each column
    possible_values = {
        'in.number_of_stories_com': [2, 1, 3, 11, 5, 6, 10, 4, 9, 20, 7, 8, 12, 14, 30],
        'in.vintage_com': [1995, 2006, 1965, 1985, 1940, 1975, 1952, 2016],
        'in.tstat_clg_sp_f..f_com': [999, 72, 73, 75, 70, 71, 74, 77, 76, 80, 79, 69],
        'in.tstat_htg_sp_f..f_com': [999, 67, 64, 69, 68, 65, 71, 66, 72, 61, 63, 70],
        'in.weekday_opening_time..hr_com': [8.25, 7., 9., 6.75, 7.75, 8., 5., 6.25, 4.25, 11., 8.5, 6., 9.25, 7.25, 9.75, 5.75, 8.75, 10.5, 7.5, 6.5, 12., 10.25, 4., 3.75, 11.5, 5.25, 3.5, 11.25, 12.25, 10., 4.5, 4.75, 9.5, 11.75, 5.5, 10.75],
        'in.weekday_operating_hours..hr_com': [9.25, 8., 11.75, 8.5, 7., 6.75, 10.5, 9., 8.75, 11.25, 10.25, 16.75, 11., 9.75, 11.5, 7.75, 8.25, 14.25, 16., 12.5, 7.25, 10., 6.25, 10.75, 14.5, 17.75, 13., 17.25, 9.5, 12.75, 6., 16.25, 13.5, 6.5, 12., 15.25, 15.5, 5.75, 16.5, 13.25, 14., 12.25, 18., 13.75, 15., 15.75, 14.75, 7.5, 17., 18.25, 17.5, 18.5, 18.75]
    }
    mapped_predictions = prediction.copy()
    # Map the predictions
    mapped_predictions_1 = map_to_closest_values(prediction[:, :6], possible_values)
    mapped_predictions_2 = encoder.inverse_transform(prediction[:, 6:])

    mapped_predictions = np.concatenate((mapped_predictions_1, mapped_predictions_2), axis=1)
    cols = ['in.number_of_stories_com', 'in.vintage_com',
       'in.tstat_clg_sp_f..f_com', 'in.tstat_htg_sp_f..f_com',
       'in.weekday_opening_time..hr_com', 'in.weekday_operating_hours..hr_com',
       'in.comstock_building_type_group_com', 'in.heating_fuel_com',
       'in.hvac_category_com', 'in.ownership_type_com', 'in.wall_construction_type_com']
    
    mapped_df = pd.DataFrame(mapped_predictions, columns=cols)
    mapped_df = mapped_df.astype({col: 'int' for col in cols[0:4]})
    period_dict = {1940:'Before 1946',  # Using an estimated middle year
                    1952:'1946 to 1959',
                    1965:'1960 to 1969',
                    1975:'1970 to 1979',
                    1985:'1980 to 1989',
                    1995:'1990 to 1999',
                    2006:'2000 to 2012',
                    2016:'2013 to 2018'
                    }
    mapped_df['in.vintage_com'] = mapped_df['in.vintage_com'].map(period_dict)
    
    return mapped_df