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


emission_data = pd.read_csv('CO2_Emissions_1960-2018.csv')
print(emission_data)

print(emission_data.describe())

emission_df = emission_data[["1971", "1981", "1991", "2001", "2016"]]
print(emission_df.describe())

emission_df = emission_df.drop(["1971", "1981"], axis=1)
print(emission_df.describe())

# Define the row and column labels
index = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
columns = ['1991', '2001', '2016']

# Create a DataFrame from the data
df_heatmap = pd.DataFrame(emission_df, index=index, columns=columns)

corr = df_heatmap.corr()

# Define custom colors for the boxes
colors = ['#FF0000', '#FFA500', '#0000FF', '#ADD8E6']  # Red, Orange, Blue, Light Blue

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', cbar=True,
            linewidths=0.5, linecolor='black', square=True)
plt.title('Correlation Heatmap with Inner Blocks')
plt.show()

pd.plotting.scatter_matrix(emission_df, figsize=(12, 12), s=5, alpha=0.8)

plt.show()

# Select the columns for clustering
df_cluster = emission_df[["1991", "2016"]]

df_cluster = df_cluster.dropna() 
df_cluster = df_cluster.reset_index()
print(df_cluster.iloc[0:15])

# remove the old index from column index before clustering
df_cluster = df_cluster.drop("index", axis=1)
print(df_cluster.iloc[0:15])

# Create and store minimum and maximum, and normalise
norm_df, min_df, max_df = ct.scaler(df_cluster)

print()
print("n  score")

# loop over number of clusters
for scluster in range(2, 10):
    
    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=scluster)

    # Fit the data, results are stored in the kmeans object
    kmeans.fit(norm_df)     # fit done on x,y pairs

    label = kmeans.labels_
    
    # extract the estimated cluster centres
    centre = kmeans.cluster_centers_

    # calculate the silhoutte score
    print(scluster, skmet.silhouette_score(df_cluster, label))
    
    scluster = 7 

# Setting up the clusterer
kmeans = cluster.KMeans(n_clusters=scluster)


kmeans.fit(norm_df)    

label = kmeans.labels_
    
# Bringing the cluster centres
centre = kmeans.cluster_centers_

centre = np.array(centre)
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

scluster = 7 

# Setting up the clusterer
kmeans = cluster.KMeans(n_clusters=scluster)


kmeans.fit(norm_df)    

label = kmeans.labels_
    
# Bringing the cluster centres
centre = kmeans.cluster_centers_

centre = np.array(centre)
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