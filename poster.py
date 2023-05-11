# -*- coding: utf-8 -*-
"""
Created on Thu May 11 00:54:11 2023

@author: Blaise Ezeokeke
"""

import pandas as pd

def read_csv_file(file_path):
    
    
    """
    Returns a Pandas DataFrame, after reading CSV files with Pandas.

    Args:
        file_path (str): The CSV file file-path.

    Returns:
        pandas.DataFrame: The CSV file data returned as a pandas DataFrame.
    """
    # Read the CSV file into a Pandas DataFrame
    data_df = pd.read_csv(file_path)

    # Return the DataFrame
    return data_df


# Read the Global CO2 emission data
co2_df = read_csv_file('co2_emissions_kt_by_country.csv')

# Preview the head of the Global CO2 emissions data
print("Global CO2 Emission Data:")
print(co2_df.head())

# Display Global CO2 emission summary statistics
print("Summary Statistics for Global CO2 Emission Data:")
print(co2_df.describe())



# Read the Global population data
pop_df = read_csv_file('world_population.csv')

# Preview the head of the Global population data
print("Global Population Data:")
print(pop_df.head())

# Display Global Population data summary statistics
print("Summary Statistics for Global Population Data:")
print(pop_df.describe())

