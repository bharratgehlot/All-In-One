# 12.	Implement hierarchical clustering to analyze and group documents based on their similarity.

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
#Also install python-docx module
import docx
import os

# Replace 'directory_path' with the path to the directory containing your .docx files
directory_path = 'CSE5thSemester'

# Initialize an empty list to store the text from all the documents
documents = []

# Iterate through each .docx file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.docx'):
        file_path = os.path.join(directory_path, filename)
        
        # Read the document and extract its text
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        
        # Append the text to the list
        documents.append(text)

'''
# Sample document dataset (replace with your dataset)
documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?"
]
'''
# Vectorize the documents using TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Calculate pairwise distances between documents
distance_matrix = pairwise_distances(tfidf_matrix, metric="cosine")

# Perform hierarchical clustering
n_clusters = 2  # Specify the number of clusters
agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity="precomputed", linkage="average")
agg_clustering.fit(distance_matrix)

# Visualize the dendrogram (optional)
'''
dendrogram(distance_matrix, 
    labels=[f'Doc {i+1}' for i in range(len(documents))],
    orientation="top",
    distance_sort="descending",
    show_leaf_counts=True,
    leaf_font_size=10,
)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Documents")
plt.ylabel("Cosine Distance")
plt.show()
'''
# Print cluster assignments
cluster_labels = agg_clustering.labels_
cluster_assignments = {i: [] for i in range(n_clusters)}
for doc_idx, cluster_label in enumerate(cluster_labels):
    cluster_assignments[cluster_label].append(f'Doc {doc_idx+1}')

for cluster_label, documents in cluster_assignments.items():
    print(f'Cluster {cluster_label + 1}:')
    print(", ".join(documents))