# -*- coding: utf-8 -*-
"""
Created on Thu May 11 00:54:11 2023

@author: Blaise Ezeokeke
"""

import sklearn.datasets as skdat
import pandas as pd
import numpy as np
import seaborn as sns
import cluster_tools as ct
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import scipy.optimize as opt
import errors as err

# Source of CO2 emission data
# https://www.kaggle.com/datasets/kkhandekar/co2-emissions-1960-2018

# Source of India GDP Data
# https://www.kaggle.com/datasets/imbikramsaha/indian-gdp


def read_emission_data(csv_file):
    """
    Read emission csv dataset using pandas.

    Args:
        csv_file (str): CSV file file Path.

    Returns:
        pandas.DataFrame: Dataframe containing the emission data.
    """
    emission_data = pd.read_csv(csv_file)
    return emission_data

# Load the emission data
emission_data = read_emission_data('CO2_Emissions_1960-2018.csv')
print(emission_data)


# Print the data summary
print(emission_data.describe())

# Select the columns of interest
emission_df = emission_data[["1971", "1981", "1991", "2001", "2016"]]
print(emission_df.describe())

# Drop unnecessary columns
emission_df = emission_df.drop(["1971", "1981"], axis=1)
print(emission_df.describe())

# Define the row and column labels
index = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
columns = ['1991', '2001', '2016']

# Create a DataFrame for the heatmap
df_heatmap = pd.DataFrame(emission_df, index=index, columns=columns)

# Calculate the correlation matrix
corr = df_heatmap.corr()

# Define custom colors for the heatmap
colors = ['#FF0000', '#FFA500', '#0000FF', '#ADD8E6']  # Red, Orange, Blue, Light Blue

# Plot the correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', cbar=True,
            linewidths=0.5, linecolor='black', square=True)
plt.title('Correlation Heatmap with Inner Blocks')
plt.show()

# Plot the scatter matrix
pd.plotting.scatter_matrix(emission_df, figsize=(12, 12), s=5, alpha=0.8)
plt.show()

# Select the columns for clustering
df_cluster = emission_df[["1991", "2016"]]

# Drop rows with missing values and reset the index
df_cluster = df_cluster.dropna().reset_index()
print(df_cluster.iloc[0:15])

# Remove the old index column
df_cluster = df_cluster.drop("index", axis=1)
print(df_cluster.iloc[0:15])

# Create and store minimum and maximum, and normalize
norm_df, min_df, max_df = ct.scaler(df_cluster)

print()
print("n  score")

# Loop over number of clusters
for scluster in range(2, 10):
    # Set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=scluster)

    # Fit the data
    kmeans.fit(norm_df)

    # Extract the estimated cluster centres
    centre = kmeans.cluster_centers_

    # Calculate the silhouette score
    label = kmeans.labels_
    print(scluster, skmet.silhouette_score(df_cluster, label))

# Reassign the number of clusters
scluster = 7

# Set up the clusterer
kmeans = cluster.KMeans(n_clusters=scluster)
kmeans.fit(norm_df)
label = kmeans.labels_
centre = kmeans.cluster_centers_
xcentre = centre[:, 0]
ycentre = centre[:, 1]



# Plot the clusters
plt.figure(figsize=(8.0, 8.0))
cluster_map = plt.cm.get_cmap('tab10')
plt.scatter(norm_df["1991"], norm_df["2016"], 10, label, marker="o", cmap=cluster_map)
plt.scatter(xcentre, ycentre, 45, "k", marker="d")
plt.xlabel("Emissions(1991)")
plt.ylabel("Emissions(2016)")
plt.show()

# DATA FITTING

# Source of Data
# https://www.kaggle.com/datasets/imbikramsaha/indian-gdp


def exponential(t, n0, g):
    """
    Calculate the exponential function with scale factor n0 and growth rate g.

    Args:
        t (float or array-like): Time values.
        n0 (float): Scale factor.
        g (float): Growth rate.

    Returns:
        float or array-like: Exponential function values.
    """
    t = t - 1990
    f = n0 * np.exp(g*t)
    return f

# Load and display the GDP data for India
India_gdp = pd.read_csv("India_GDP_Data.csv")
print(India_gdp)

# Select the required columns
India_gdp = India_gdp[["Year", "Per_Capita_in_USD"]]
print(India_gdp)

# Plot the GDP data
India_gdp.plot("Year", "Per_Capita_in_USD")

# Add titles, labels, and legend
plt.title('India GDP Per Capita Growth')
plt.xlabel('GDP Per Capita')
plt.ylabel('Year')
plt.legend()

# Display the plot
plt.show()

def convert_column_to_numeric(India_gdp, Year):
    """
    Converts a column in a DataFrame to numeric data type.

    Args:
        df (pandas.DataFrame): DataFrame containing the data.
        column (str): Name of the column to convert.

    Returns:
        pandas.DataFrame: DataFrame with the converted column.
    """
    India_gdp[Year] = pd.to_numeric(India_gdp[Year])
    return India_gdp



# Fit the exponential curve to the data
param, covar = opt.curve_fit(exponential, India_gdp["Year"], India_gdp["Per_Capita_in_USD"], p0=(1.2e12, 0.03))

# Print the results
print("GDP 1990:", param[0] / 1e9)
print("Growth rate:", param[1])

# Plot the fitted exponential curve and the actual data
plt.figure()
plt.plot(India_gdp["Year"], exponential(India_gdp["Year"], 1.2e12, 0.03), label="Trial Fit")
plt.plot(India_gdp["Year"], India_gdp["Per_Capita_in_USD"])

# Add titles, labels, and legend
plt.title("GDP Per Capita")
plt.xlabel("Year")
plt.ylabel("GDP Per Capita")
plt.legend()

# Display the plot
plt.show()

# Add the fitted curve to the DataFrame
India_gdp["fit"] = exponential(India_gdp["Year"], *param)

# Plot the actual data and the fitted curve
India_gdp.plot("Year", ["Per_Capita_in_USD", "fit"])
plt.title("Exponential Fit")
plt.xlabel("Year")
plt.ylabel("GDP Per Capita")
plt.legend()

# Display the plot
plt.show()

# Generate forecast for future years
year = np.arange(1960, 2031)
forecast = exponential(year, *param)

# Plot the actual data, forecast, and confidence interval
plt.figure()
plt.plot(India_gdp["Year"], India_gdp["Per_Capita_in_USD"], label="GDP")
plt.plot(year, forecast, label="Forecast")

# Add the confidence interval as a shaded area
low, up = err.err_ranges(year, exponential, param, np.sqrt(np.diag(covar)))
plt.fill_between(year, low, up, color="yellow", alpha=0.7)

# Add titles, labels, and legend
plt.title("GDP Forecast till 2030")
plt.xlabel("Year")
plt.ylabel("GDP Per Capita")
plt.legend()

# Display the plot
plt.show()
