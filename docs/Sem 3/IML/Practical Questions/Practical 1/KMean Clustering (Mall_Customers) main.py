
#segment customer data into different groups using K-means clustering

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load your customer dataset (replace 'customer_data.csv' with your dataset)
data = pd.read_csv('Mall_Customers.csv')

# Select relevant features for clustering (e.g., age and income)
# Adjust the feature selection based on your dataset
features = data[['age', 'income']]

# Standardize the feature data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

# Determine the optimal number of clusters using the elbow method or silhouette score
# Uncomment one of the following methods

# Method 1: Elbow method to find the optimal number of clusters
# wcss = []  # Within-cluster sum of squares
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
#     kmeans.fit(scaled_data)
#     wcss.append(kmeans.inertia_)
# plt.plot(range(1, 11), wcss)
# plt.title('Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()

# Method 2: Silhouette score to find the optimal number of clusters
# silhouette_scores = []
# for i in range(2, 11):
#     kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
#     kmeans.fit(scaled_data)
#     silhouette_scores.append(silhouette_score(scaled_data, kmeans.labels_))
# plt.plot(range(2, 11), silhouette_scores)
# plt.title('Silhouette Score Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('Silhouette Score')
# plt.show()

# Choose the optimal number of clusters based on the above analysis
n_clusters = 3  # Adjust this based on your analysis

# Perform K-means clustering
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Add cluster labels to the original dataset
data['cluster'] = clusters

# Visualize the clusters (for 2D data)
plt.scatter(data['age'], data['income'], c=data['cluster'], cmap='rainbow')
plt.title('Customer Segmentation')
plt.xlabel('Age')
plt.ylabel('Income')
plt.show()

# Optionally, you can explore and analyze the characteristics of each cluster further
for cluster_id in range(n_clusters):
    cluster_data = data[data['cluster'] == cluster_id]
    print(f"Cluster {cluster_id}:")
    print(cluster_data.describe())