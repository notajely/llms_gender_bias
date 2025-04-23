#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Representational Distortion Analysis Script

This script identifies occupations with significant representational distortion,
where the model's gender projection scores poorly correlate with real-world female ratios.
It calculates distortion scores and identifies the most distorted occupations.

Author: Jiayu Shen
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Create results directory if it doesn't exist
RESULTS_DIR = './results/distortion_analysis'
os.makedirs(RESULTS_DIR, exist_ok=True)

# File paths
OCCUPATION_DATA_PATH = './data/occupation_gender_data.csv'
PROJECTION_SCORES_PATH = './results/occupation_gender_bias_results.csv'

def load_occupation_data(file_path=OCCUPATION_DATA_PATH):
    """
    Load occupation data with gender ratios from BLS.

    Args:
        file_path (str): Path to the occupation data CSV file

    Returns:
        pd.DataFrame: DataFrame containing occupation data
    """
    df = pd.read_csv(file_path)
    print(f"Loaded occupation data with {df.shape[0]} occupations")
    return df

def load_projection_scores(file_path=PROJECTION_SCORES_PATH):
    """
    Load gender projection scores.

    Args:
        file_path (str): Path to the projection scores CSV file

    Returns:
        pd.DataFrame: DataFrame containing projection scores
    """
    try:
        df = pd.read_csv(file_path)
        # Check if the file already has projection scores
        if 'projection_score' in df.columns:
            print(f"Loaded gender projection scores for {df.shape[0]} occupations")
            return df[['occupation', 'projection_score']]
        else:
            print(f"Loaded occupation analysis results for {df.shape[0]} occupations")
            # If no projection_score column, use the similarity_bias or calculate from male/female similarity
            if 'similarity_bias' in df.columns:
                # Rename similarity_bias to projection_score for consistency
                df_result = df[['occupation', 'similarity_bias']].copy()
                df_result.rename(columns={'similarity_bias': 'projection_score'}, inplace=True)
                return df_result
            elif all(col in df.columns for col in ['male_similarity', 'female_similarity']):
                # Calculate bias as male_similarity - female_similarity
                df_result = df[['occupation', 'male_similarity', 'female_similarity']].copy()
                df_result['projection_score'] = df_result['male_similarity'] - df_result['female_similarity']
                return df_result[['occupation', 'projection_score']]
            else:
                print("Error: Could not find projection_score or components to calculate it")
                return None
    except Exception as e:
        print(f"Error loading projection scores: {e}")
        return None

def merge_datasets(occupation_df, projection_df):
    """
    Merge occupation data with projection scores.

    Args:
        occupation_df (pd.DataFrame): DataFrame containing occupation data
        projection_df (pd.DataFrame): DataFrame containing projection scores

    Returns:
        pd.DataFrame: Merged DataFrame
    """
    # Merge datasets
    merged_df = pd.merge(occupation_df, projection_df, on='occupation', how='inner')
    print(f"Merged dataset contains {merged_df.shape[0]} occupations")

    # Clean up stereotype labels (remove proxy labels if present)
    if 'bls_label' in merged_df.columns:
        merged_df['bls_label'] = merged_df['bls_label'].astype(str).str.replace(' (proxy)', '', regex=False)

    # Filter out unknown labels
    if 'bls_label' in merged_df.columns:
        original_count = len(merged_df)
        merged_df = merged_df[~merged_df['bls_label'].isin(['unknown', 'unknown_insufficient_data'])].reset_index(drop=True)
        print(f"Filtered out occupations with unknown labels. {original_count - len(merged_df)} rows removed.")

    return merged_df

def calculate_distortion_scores(df):
    """
    Calculate distortion scores for each occupation.

    The distortion score measures how much the model's projection score deviates
    from what would be expected based on real-world gender ratios.

    Args:
        df (pd.DataFrame): Merged DataFrame with occupation data and projection scores

    Returns:
        pd.DataFrame: DataFrame with distortion scores
    """
    print("Calculating distortion scores...")

    # Create a copy of the DataFrame
    df_distortion = df.copy()

    # Ensure we have the necessary columns
    required_columns = ['bls_female', 'projection_score']
    if not all(col in df_distortion.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df_distortion.columns]
        print(f"Error: Missing required columns: {missing}")
        return None

    # Create a linear regression model to predict projection scores from female ratios
    X = df_distortion['bls_female'].values.reshape(-1, 1)
    y = df_distortion['projection_score'].values

    # Fit the model
    model = LinearRegression()
    model.fit(X, y)

    # Get the predicted projection scores
    df_distortion['expected_projection'] = model.predict(X)

    # Calculate the residuals (actual - expected)
    df_distortion['residual'] = df_distortion['projection_score'] - df_distortion['expected_projection']

    # Calculate the standardized residuals (z-scores)
    df_distortion['std_residual'] = stats.zscore(df_distortion['residual'])

    # Calculate the absolute distortion (magnitude of standardized residual)
    df_distortion['abs_distortion'] = np.abs(df_distortion['std_residual'])

    # Determine distortion direction
    df_distortion['distortion_direction'] = np.where(
        df_distortion['std_residual'] > 0,
        'male-biased',  # More masculine than expected based on female ratio
        'female-biased'  # More feminine than expected based on female ratio
    )

    # Calculate regression statistics
    r_squared = model.score(X, y)
    slope = model.coef_[0]
    intercept = model.intercept_

    print(f"Regression model: projection_score = {slope:.4f} * bls_female + {intercept:.4f}")
    print(f"R-squared: {r_squared:.4f}")

    return df_distortion

def identify_distorted_occupations(df_distortion, threshold=1.5):
    """
    Identify occupations with significant distortion.

    Args:
        df_distortion (pd.DataFrame): DataFrame with distortion scores
        threshold (float): Z-score threshold for significant distortion

    Returns:
        pd.DataFrame: DataFrame with significantly distorted occupations
    """
    print(f"Identifying occupations with distortion above {threshold} standard deviations...")

    # Filter occupations with absolute standardized residual above threshold
    df_significant = df_distortion[df_distortion['abs_distortion'] > threshold].copy()

    # Sort by absolute distortion (descending)
    df_significant = df_significant.sort_values('abs_distortion', ascending=False)

    print(f"Found {len(df_significant)} occupations with significant distortion")

    return df_significant

def create_distortion_visualization(df_distortion, df_significant):
    """
    Create visualizations of distortion analysis.

    Args:
        df_distortion (pd.DataFrame): DataFrame with all distortion scores
        df_significant (pd.DataFrame): DataFrame with significantly distorted occupations

    Returns:
        None
    """
    print("Creating visualizations...")

    # Set style
    sns.set(style="whitegrid")

    # 1. Scatter plot with regression line and highlighted distorted occupations
    plt.figure(figsize=(12, 10))

    # Plot all occupations
    sns.scatterplot(
        x='bls_female',
        y='projection_score',
        hue='bls_label',
        data=df_distortion,
        alpha=0.6,
        s=80,
        palette='coolwarm_r'
    )

    # Add regression line
    sns.regplot(
        x='bls_female',
        y='projection_score',
        data=df_distortion,
        scatter=False,
        color='black',
        line_kws={'linestyle': '--', 'linewidth': 1}
    )

    # Highlight significantly distorted occupations
    plt.scatter(
        df_significant['bls_female'],
        df_significant['projection_score'],
        s=120,
        facecolors='none',
        edgecolors='red',
        linewidth=2
    )

    # Add labels for significantly distorted occupations
    for _, row in df_significant.iterrows():
        plt.annotate(
            row['occupation'],
            (row['bls_female'], row['projection_score']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            weight='bold'
        )

    # Set labels and title
    plt.title('Occupation Gender Distortion Analysis', fontsize=16)
    plt.xlabel('Female Ratio in Occupation (BLS data)', fontsize=14)
    plt.ylabel('Gender Projection Score (higher = more masculine)', fontsize=14)

    # Add legend
    plt.legend(title='BLS Label', fontsize=10, title_fontsize=12)

    # Save figure
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/distortion_analysis_scatter.png', dpi=300)
    plt.savefig(f'{RESULTS_DIR}/distortion_analysis_scatter.pdf')
    plt.close()

    # 2. Bar plot of the most distorted occupations
    plt.figure(figsize=(14, 10))

    # Get top 20 most distorted occupations (or all if fewer than 20)
    top_n = min(20, len(df_significant))
    df_top = df_significant.head(top_n)

    # Create bar plot
    bars = sns.barplot(
        x='std_residual',
        y='occupation',
        hue='distortion_direction',
        palette={'male-biased': '#95B3D7', 'female-biased': '#FFB598'},
        data=df_top.sort_values('std_residual')
    )

    # Add vertical line at x=0
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

    # Set labels and title
    plt.title('Top Occupations with Significant Representational Distortion', fontsize=16)
    plt.xlabel('Standardized Residual (Z-score)', fontsize=14)
    plt.ylabel('Occupation', fontsize=14)

    # Add legend
    plt.legend(title='Distortion Direction', fontsize=10, title_fontsize=12)

    # Save figure
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/top_distorted_occupations.png', dpi=300)
    plt.savefig(f'{RESULTS_DIR}/top_distorted_occupations.pdf')
    plt.close()

    # 3. Distribution of distortion by stereotype category
    plt.figure(figsize=(12, 8))

    # Create box plot
    sns.boxplot(
        x='bls_label',
        y='std_residual',
        data=df_distortion,
        palette='coolwarm_r',
        order=['male-stereotyped', 'neutral', 'female-stereotyped']
    )

    # Add horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Set labels and title
    plt.title('Distribution of Distortion by Stereotype Category', fontsize=16)
    plt.xlabel('Stereotype Category', fontsize=14)
    plt.ylabel('Standardized Residual (Z-score)', fontsize=14)

    # Save figure
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/distortion_by_category.png', dpi=300)
    plt.savefig(f'{RESULTS_DIR}/distortion_by_category.pdf')
    plt.close()

def save_results(df_distortion, df_significant):
    """
    Save analysis results to CSV files.

    Args:
        df_distortion (pd.DataFrame): DataFrame with all distortion scores
        df_significant (pd.DataFrame): DataFrame with significantly distorted occupations

    Returns:
        None
    """
    print("Saving results...")

    # Save all distortion scores
    df_distortion.to_csv(f'{RESULTS_DIR}/all_occupation_distortion_scores.csv', index=False)

    # Save significantly distorted occupations
    df_significant.to_csv(f'{RESULTS_DIR}/significant_distortion_occupations.csv', index=False)

    print(f"Results saved to {RESULTS_DIR}/")

def main():
    """Main function to run the analysis."""
    print("Starting representational distortion analysis...")

    # Load data
    occupation_df = load_occupation_data()
    projection_df = load_projection_scores()

    if projection_df is None:
        print("Error: Could not load projection scores. Exiting.")
        return

    # Merge datasets
    merged_df = merge_datasets(occupation_df, projection_df)

    # Calculate distortion scores
    df_distortion = calculate_distortion_scores(merged_df)

    if df_distortion is None:
        print("Error: Could not calculate distortion scores. Exiting.")
        return

    # Identify significantly distorted occupations
    df_significant = identify_distorted_occupations(df_distortion)

    # Create visualizations
    create_distortion_visualization(df_distortion, df_significant)

    # Save results
    save_results(df_distortion, df_significant)

    # Print summary
    print("\nDistortion Analysis Summary:")
    print(f"Total occupations analyzed: {len(df_distortion)}")
    print(f"Occupations with significant distortion: {len(df_significant)}")

    # Print top 10 most distorted occupations
    print("\nTop 10 most distorted occupations:")
    top10 = df_significant.head(10)
    for _, row in top10.iterrows():
        direction = "more masculine" if row['distortion_direction'] == 'male-biased' else "more feminine"
        print(f"  {row['occupation']}: {row['abs_distortion']:.2f} SD {direction} than expected")

    print("\nAnalysis complete. Results saved to the results directory.")

if __name__ == "__main__":
    main()
