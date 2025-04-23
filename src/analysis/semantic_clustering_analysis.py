#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Perform semantic clustering analysis on occupation embeddings.

This script:
1. Loads the occupation embeddings
2. Performs KMeans clustering with k=2
3. Applies PCA to reduce the embeddings to 2 dimensions
4. Creates a visualization of the clusters
5. Analyzes the clustering results
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import csv

# Configuration
EMBEDDINGS_PATH = 'results/semantic_clustering/occupation_embeddings.npy'
NAMES_PATH = 'results/semantic_clustering/occupation_names.pkl'
OCCUPATION_LIST_PATH = 'data/bias_in_bia_occupation_list.csv'
OUTPUT_DIR = 'results/semantic_clustering'
NUM_CLUSTERS = 2
RANDOM_SEED = 42

def load_data():
    """Load the occupation embeddings and names."""
    print(f"Loading embeddings from {EMBEDDINGS_PATH}...")
    embeddings = np.load(EMBEDDINGS_PATH)
    
    print(f"Loading occupation names from {NAMES_PATH}...")
    with open(NAMES_PATH, 'rb') as f:
        occupation_names = pickle.load(f)
    
    return embeddings, occupation_names

def load_occupation_list():
    """Load the occupation list from CSV file."""
    occupation_list = []
    with open(OCCUPATION_LIST_PATH, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            occupation_list.append((int(row[0]), row[1]))
    return occupation_list

def perform_kmeans_clustering(embeddings):
    """
    Perform KMeans clustering on the embeddings.
    
    Args:
        embeddings: Numpy array of embeddings
        
    Returns:
        KMeans model and cluster labels
    """
    print(f"Performing KMeans clustering with k={NUM_CLUSTERS}...")
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=RANDOM_SEED, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(embeddings, labels)
    print(f"Silhouette score: {silhouette_avg:.4f}")
    
    return kmeans, labels

def perform_pca(embeddings):
    """
    Perform PCA on the embeddings.
    
    Args:
        embeddings: Numpy array of embeddings
        
    Returns:
        PCA model and reduced embeddings
    """
    print("Performing PCA...")
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # Print explained variance
    explained_variance = pca.explained_variance_ratio_
    print(f"Explained variance: {explained_variance[0]:.4f}, {explained_variance[1]:.4f}")
    print(f"Total explained variance: {sum(explained_variance):.4f}")
    
    return pca, reduced_embeddings

def create_visualization(reduced_embeddings, labels, occupation_names):
    """
    Create a visualization of the clusters.
    
    Args:
        reduced_embeddings: PCA-reduced embeddings
        labels: Cluster labels
        occupation_names: List of occupation names
    """
    print("Creating visualization...")
    plt.figure(figsize=(12, 10))
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create scatter plot
    scatter = plt.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        c=labels,
        cmap='coolwarm',
        s=100,
        alpha=0.7
    )
    
    # Add occupation labels
    for i, occupation in enumerate(occupation_names):
        plt.annotate(
            occupation,
            (reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=10,
            alpha=0.8
        )
    
    # Add title and labels
    plt.title('PCA of Contextualized Occupation Embeddings', fontsize=16)
    plt.xlabel('First Principal Component', fontsize=12)
    plt.ylabel('Second Principal Component', fontsize=12)
    
    # Add legend
    legend = plt.legend(
        *scatter.legend_elements(),
        title="Clusters",
        loc="upper right"
    )
    plt.gca().add_artist(legend)
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, 'pca_visualization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    
    # Close figure
    plt.close()

def analyze_clusters(labels, occupation_names, occupation_list):
    """
    Analyze the clustering results.
    
    Args:
        labels: Cluster labels
        occupation_names: List of occupation names
        occupation_list: List of (id, name) tuples for occupations
    """
    print("Analyzing clusters...")
    
    # Create a mapping from occupation name to profession ID
    occupation_to_id = {name: id for id, name in occupation_list}
    
    # Create a DataFrame with the results
    results = []
    for i, occupation in enumerate(occupation_names):
        results.append({
            'occupation': occupation,
            'cluster': int(labels[i]),
            'profession_id': occupation_to_id.get(occupation, -1)
        })
    
    df_results = pd.DataFrame(results)
    
    # Count occupations in each cluster
    cluster_counts = df_results['cluster'].value_counts().sort_index()
    print("\nCluster counts:")
    for cluster, count in cluster_counts.items():
        print(f"Cluster {cluster}: {count} occupations")
    
    # List occupations in each cluster
    print("\nOccupations in each cluster:")
    for cluster in range(NUM_CLUSTERS):
        occupations_in_cluster = df_results[df_results['cluster'] == cluster]['occupation'].tolist()
        print(f"\nCluster {cluster}:")
        for occupation in occupations_in_cluster:
            print(f"  - {occupation}")
    
    # Save results
    output_path = os.path.join(OUTPUT_DIR, 'clustering_results.csv')
    df_results.to_csv(output_path, index=False)
    print(f"Saved clustering results to {output_path}")

def main():
    """Main function to perform clustering analysis."""
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    embeddings, occupation_names = load_data()
    occupation_list = load_occupation_list()
    
    # Perform KMeans clustering
    kmeans, labels = perform_kmeans_clustering(embeddings)
    
    # Perform PCA
    pca, reduced_embeddings = perform_pca(embeddings)
    
    # Create visualization
    create_visualization(reduced_embeddings, labels, occupation_names)
    
    # Analyze clusters
    analyze_clusters(labels, occupation_names, occupation_list)

if __name__ == "__main__":
    main()
