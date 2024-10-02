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

def inverse_process_com(prediction, encoder):
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

def inverse_process_res(prediction, encoder):
    prediction = prediction.cpu().detach().numpy()

    # Possible values for each column
    possible_values = {
        'in.bedrooms_res': [3, 2, 1, 4, 5],
        'in.cooling_setpoint_res': [68, 75, 72, 70, 78, 80, 60, 65, 76, 67, 62],
        'in.heating_setpoint_res': [75, 72, 68, 70, 62, 55, 65, 78, 76, 67, 60, 80],
        'in.geometry_floor_area_res': [1749,  874, 1249,  624, 2249, 4000, 2749, 3499,  249],
        'in.income_res': [109999,  12499,  64999,  89999,  42499,  22499,  47499, 169999, 129999,
                            200000,  37499,  54999,  32499,  10000, 189999,  74999,  17499,  27499,
                            149999],
        'in.vintage_res': [1940, 1975, 1985, 1945, 1995, 1955, 1965, 2005, 2015]
    }
    mapped_predictions = prediction.copy()
    # Map the predictions
    mapped_predictions_1 = map_to_closest_values(prediction[:, :6], possible_values)
    mapped_predictions_2 = encoder.inverse_transform(prediction[:, 6:])

    mapped_predictions = np.concatenate((mapped_predictions_1, mapped_predictions_2), axis=1)
    cols = ['in.bedrooms_res', 'in.cooling_setpoint_res', 'in.heating_setpoint_res',
            'in.geometry_floor_area_res', 'in.income_res', 'in.vintage_res',
            'in.geometry_building_type_recs_res', 'in.geometry_foundation_type_res', 
            'in.geometry_wall_type_res', 'in.heating_fuel_res', 'in.roof_material_res',
            'in.tenure_res', 'in.vacancy_status_res']
    
    mapped_df = pd.DataFrame(mapped_predictions, columns=cols)

    vintage_mapping = {1940:'<1940', 1945:'1940s', 1955:'1950s', 1965:'1960s',
                           1975:'1970s', 1985:'1980s', 1995:'1990s', 2005:'2000s', 2015:'2010s'}
    mapped_df['in.vintage_res'] = mapped_df['in.vintage_res'].map(vintage_mapping)
    mapped_df['in.cooling_setpoint_res'] = mapped_df['in.cooling_setpoint_res'].apply(lambda x: str(x)+"F").astype(str)
    mapped_df['in.heating_setpoint_res'] = mapped_df['in.heating_setpoint_res'].apply(lambda x: str(x)+"F").astype(str)
    mapped_df['in.bedrooms_res'] = mapped_df['in.bedrooms_res'].astype(str)

    income_mapping = {109999: '100000-119999', 12499: '10000-14999', 64999: '60000-69999', 89999: '80000-99999', 
                      42499: '40000-44999', 22499: '20000-24999', 47499: '45000-49999', 169999: '160000-179999', 
                      129999: '120000-139999', 200000: '200000+', 37499: '35000-39999', 54999: '50000-59999', 
                      32499: '30000-34999', 10000: '<10000', 189999: '180000-199999', 74999: '70000-79999', 
                      17499: '15000-19999', 27499: '25000-29999', 149999: '140000-159999'}
    mapped_df['in.income_res'] = mapped_df['in.income_res'].map(income_mapping)

    geometry_mapping = {1749: '1500-1999', 874: '750-999', 1249: '1000-1499', 624: '500-749', 2249: '2000-2499',
                        4000: '4000+', 2749: '2500-2999', 3499: '3000-3999', 249: '0-499'}
    mapped_df['in.geometry_floor_area_res'] = mapped_df['in.geometry_floor_area_res'].map(geometry_mapping)

    return mapped_df