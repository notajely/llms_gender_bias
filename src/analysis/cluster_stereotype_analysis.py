#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze the relationship between semantic clusters and gender stereotypes.

This script:
1. Loads the clustering results
2. Assigns gender stereotype labels to occupations based on BLS data
3. Analyzes the relationship between clusters and gender stereotypes
4. Creates visualizations of the results
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA

# Configuration
BIB_BIAS_CLUSTER_PATH = 'data/bib_bias_cluster_analysis.csv'
EMBEDDINGS_PATH = 'results/semantic_clustering/occupation_embeddings.npy'
OCCUPATION_NAMES_PATH = 'results/semantic_clustering/occupation_names.pkl'
OUTPUT_DIR = 'results/semantic_clustering'

# PCA parameters
GENERATE_PCA_COORDS = True  # Whether to generate PCA coordinates if not available

def load_data():
    """Load the BiB bias cluster analysis data."""
    print(f"Loading BiB bias cluster analysis data from {BIB_BIAS_CLUSTER_PATH}...")
    try:
        df = pd.read_csv(BIB_BIAS_CLUSTER_PATH)
        print(f"Loaded data with {len(df)} occupations")

        # Print column information
        print(f"Columns: {df.columns.tolist()}")

        return df
    except FileNotFoundError:
        print(f"Error: BiB bias cluster analysis data not found at {BIB_BIAS_CLUSTER_PATH}")
        print("Please make sure the file exists")
        sys.exit(1)

def generate_pca_coordinates(df):
    """
    Generate PCA coordinates for the occupations.

    Args:
        df: DataFrame with occupation data

    Returns:
        DataFrame with added PCA coordinates
    """
    print("Generating PCA coordinates...")

    # Check if we already have x and y coordinates
    if 'x' in df.columns and 'y' in df.columns:
        print("PCA coordinates already exist in the data")
        return df

    # Try to load embeddings
    try:
        # Load occupation embeddings
        print(f"Loading occupation embeddings from {EMBEDDINGS_PATH}...")
        embeddings = np.load(EMBEDDINGS_PATH)

        # Load occupation names
        print(f"Loading occupation names from {OCCUPATION_NAMES_PATH}...")
        with open(OCCUPATION_NAMES_PATH, 'rb') as f:
            occupation_names = pickle.load(f)

        print(f"Loaded embeddings for {len(occupation_names)} occupations")

        # Create a mapping from occupation name to embedding
        occupation_to_embedding = {}
        for i, name in enumerate(occupation_names):
            occupation_to_embedding[name] = embeddings[i]

        # Check if we have embeddings for all occupations in the dataframe
        missing_occupations = []
        for occupation in df['occupation']:
            if occupation not in occupation_to_embedding:
                missing_occupations.append(occupation)

        if missing_occupations:
            print(f"Warning: Missing embeddings for {len(missing_occupations)} occupations")
            print(f"Missing occupations: {missing_occupations}")
            print("Will use random coordinates for these occupations")

            # Generate random embeddings for missing occupations
            np.random.seed(42)  # For reproducibility
            for occupation in missing_occupations:
                occupation_to_embedding[occupation] = np.random.randn(embeddings.shape[1])

        # Create a matrix of embeddings in the same order as the dataframe
        occupation_embeddings = np.array([occupation_to_embedding[occupation] for occupation in df['occupation']])

        # Perform PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(occupation_embeddings)

        # Add PCA coordinates to the dataframe
        df['x'] = pca_result[:, 0]
        df['y'] = pca_result[:, 1]

        # Print explained variance
        explained_variance = pca.explained_variance_ratio_
        print(f"Explained variance: {explained_variance[0]:.4f}, {explained_variance[1]:.4f}")
        print(f"Total explained variance: {sum(explained_variance):.4f}")

        return df
    except Exception as e:
        print(f"Warning: Could not generate PCA coordinates: {e}")
        print("Will use predefined coordinates instead")

        # Generate predefined coordinates based on cluster
        # Cluster 0 on the left, Cluster 1 on the right
        df['x'] = np.where(df['cluster'] == 0, -0.5, 0.5) + np.random.randn(len(df)) * 0.1
        df['y'] = np.random.randn(len(df)) * 0.3

        return df



def analyze_stereotype_distribution(df):
    """
    Analyze the distribution of gender stereotypes in each cluster.

    Args:
        df: DataFrame with clustering results and stereotype labels
    """
    print("Analyzing stereotype distribution...")

    # Count stereotypes in each cluster
    stereotype_counts = pd.crosstab(df['cluster'], df['stereotype'])
    print("\nStereotype distribution by cluster:")
    print(stereotype_counts)

    # Calculate percentages
    stereotype_percentages = stereotype_counts.div(stereotype_counts.sum(axis=1), axis=0) * 100
    print("\nStereotype percentages by cluster:")
    print(stereotype_percentages.round(2))

    # Create visualization
    plt.figure(figsize=(10, 6))

    # Set style
    sns.set_style("whitegrid")

    # Create stacked bar chart
    ax = stereotype_percentages.plot(
        kind='bar',
        stacked=True,
        colormap='coolwarm',
        figsize=(10, 6)
    )

    # Add title and labels
    plt.title('Gender Stereotype Distribution by Cluster', fontsize=16)
    plt.xlabel('Cluster', fontsize=12)
    plt.ylabel('Percentage', fontsize=12)

    # Add legend
    plt.legend(title='Gender Stereotype')

    # Save figure
    output_path = os.path.join(OUTPUT_DIR, 'stereotype_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved stereotype distribution visualization to {output_path}")

    # Close figure
    plt.close()

def create_cluster_stereotype_visualization(df):
    """
    Create a visualization of clusters and stereotypes using PCA coordinates.

    Args:
        df: DataFrame with clustering results, stereotype labels, and PCA coordinates
    """
    print("Creating cluster-stereotype visualization...")

    # Check if we have x and y coordinates in the dataframe
    if 'x' not in df.columns or 'y' not in df.columns:
        print("Error: PCA coordinates (x, y) not found in the dataframe")
        print("This should not happen as coordinates should have been generated")
        return

    # Create figure
    plt.figure(figsize=(14, 12))

    # Set style
    sns.set_style("whitegrid")

    # Define colors for stereotypes
    stereotype_colors = {
        'male-stereotyped': '#1f77b4',      # Blue
        'male-stereotyped (proxy)': '#1f77b4',  # Blue (same as male-stereotyped)
        'neutral': '#7f7f7f',               # Gray
        'neutral (proxy)': '#7f7f7f',       # Gray (same as neutral)
        'female-stereotyped': '#d62728',    # Red
        'excluded': '#8c564b'               # Brown
    }

    # Create scatter plot
    for stereotype in df['stereotype'].unique():
        if stereotype == 'excluded':
            continue  # Skip excluded occupations

        mask = df['stereotype'] == stereotype
        if mask.any():
            color = stereotype_colors.get(stereotype, '#2ca02c')  # Default to green if not in colors
            plt.scatter(
                df.loc[mask, 'x'],
                df.loc[mask, 'y'],
                c=color,
                label=stereotype,
                s=120,
                alpha=0.7
            )

    # Add occupation labels
    for i, row in df.iterrows():
        if row['stereotype'] == 'excluded':
            continue  # Skip excluded occupations

        plt.annotate(
            row['occupation'],
            (row['x'], row['y']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=10,
            alpha=0.8
        )

    # Add cluster boundaries or centers
    for cluster in sorted(df['cluster'].unique()):
        mask = df['cluster'] == cluster
        if mask.any():
            center_x = df.loc[mask, 'x'].mean()
            center_y = df.loc[mask, 'y'].mean()
            plt.scatter(
                center_x,
                center_y,
                marker='X',
                s=250,
                edgecolor='black',
                linewidth=2,
                alpha=0.8,
                label=f'Cluster {cluster} Center'
            )

            # Draw a circle around the cluster center
            circle = mpatches.Circle((center_x, center_y), 0.5, fill=False, linestyle='--',
                               color='black', alpha=0.3)
            plt.gca().add_patch(circle)

    # Add title and labels
    plt.title('PCA of Occupation Embeddings with Gender Stereotypes', fontsize=18)
    plt.xlabel('First Principal Component', fontsize=14)
    plt.ylabel('Second Principal Component', fontsize=14)

    # Add legend with a better layout
    plt.legend(title="Gender Stereotype", loc="upper right", fontsize=12, title_fontsize=14)

    # Save figure
    output_path = os.path.join(OUTPUT_DIR, 'cluster_stereotype_visualization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved cluster-stereotype visualization to {output_path}")

    # Close figure
    plt.close()

def evaluate_cluster_stereotype_relationship(df):
    """
    Evaluate the relationship between clusters and gender stereotypes.

    Args:
        df: DataFrame with clustering results and stereotype labels
    """
    print("Evaluating cluster-stereotype relationship...")

    # Filter out unknown stereotypes if present
    if 'unknown' in df['stereotype'].values:
        df_known = df[df['stereotype'] != 'unknown'].copy()
    else:
        df_known = df.copy()

    # Check if we have enough data
    if len(df_known) < 5:
        print("Not enough data with known stereotypes to evaluate relationship")
        return

    # Create a mapping from stereotype to cluster
    stereotype_to_cluster = {}
    for _, row in df_known.iterrows():
        stereotype = row['stereotype']
        cluster = row['cluster']
        if stereotype not in stereotype_to_cluster:
            stereotype_to_cluster[stereotype] = []
        stereotype_to_cluster[stereotype].append(cluster)

    # Print the distribution of clusters for each stereotype
    print("\nCluster distribution by stereotype:")
    for stereotype, clusters in stereotype_to_cluster.items():
        cluster_counts = pd.Series(clusters).value_counts().sort_index()
        print(f"\n{stereotype}:")
        for cluster, count in cluster_counts.items():
            percentage = count / len(clusters) * 100
            print(f"  Cluster {cluster}: {count} occupations ({percentage:.1f}%)")

    # Create a contingency table
    contingency_table = pd.crosstab(df_known['stereotype'], df_known['cluster'])
    print("\nContingency Table:")
    print(contingency_table)

    # Save results
    output_path = os.path.join(OUTPUT_DIR, 'cluster_stereotype_evaluation.csv')
    contingency_table.to_csv(output_path)
    print(f"Saved cluster-stereotype evaluation to {output_path}")

def main():
    """Main function to analyze stereotype distribution."""
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    df = load_data()

    # Check if required columns exist
    required_columns = ['occupation', 'cluster', 'bls_label']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        print(f"Available columns: {df.columns.tolist()}")
        sys.exit(1)

    # Rename bls_label to stereotype for consistency
    df = df.rename(columns={'bls_label': 'stereotype'})

    # Generate PCA coordinates if needed
    if GENERATE_PCA_COORDS and ('x' not in df.columns or 'y' not in df.columns):
        df = generate_pca_coordinates(df)

    # Save a copy of the data with PCA coordinates
    output_path = os.path.join(OUTPUT_DIR, 'bib_clustering_results.csv')
    df.to_csv(output_path, index=False)
    print(f"Saved BiB clustering results to {output_path}")

    # Analyze stereotype distribution
    analyze_stereotype_distribution(df)

    # Create visualization
    create_cluster_stereotype_visualization(df)

    # Evaluate relationship
    evaluate_cluster_stereotype_relationship(df)

if __name__ == "__main__":
    main()
