#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive analysis of the Bias in Bios dataset.

This script:
1. Loads the Bias in Bios dataset
2. Analyzes the overall dataset structure
3. Examines the distribution of occupations
4. Analyzes gender distribution by occupation
5. Provides text length statistics
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter

# Configuration
RAW_DATA_PATH = 'data/raw/bios_train.pkl'
OCCUPATION_LIST_PATH = 'data/bias_in_bia_occupation_list.csv'
OUTPUT_DIR = 'results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_occupation_list():
    """Load the occupation list from CSV file."""
    occupation_list = []
    with open(OCCUPATION_LIST_PATH, 'r') as f:
        for line in f:
            code, name = line.strip().split(',')
            occupation_list.append((int(code), name))
    return occupation_list

def load_bib_data():
    """Load the Bias in Bios dataset."""
    print(f"Loading BiB data from {RAW_DATA_PATH}...")
    with open(RAW_DATA_PATH, 'rb') as f:
        data = pickle.load(f)
    return data

def analyze_dataset_structure(data):
    """Analyze the overall structure of the dataset."""
    print("\n=== Dataset Structure ===")
    print(f"Data type: {type(data)}")
    print(f"Total number of samples: {len(data)}")
    print(f"Columns: {data.columns.tolist()}")
    
    # Sample data
    print("\nSample data (first 3 rows):")
    print(data.head(3))
    
    # Memory usage
    memory_usage = data.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"\nMemory usage: {memory_usage:.2f} MB")

def analyze_occupation_distribution(data, occupation_map):
    """Analyze the distribution of occupations in the dataset."""
    print("\n=== Occupation Distribution ===")
    
    # Count samples per occupation
    profession_counts = data['profession'].value_counts().sort_index()
    
    # Create a DataFrame for better visualization
    df_counts = pd.DataFrame({
        'profession_code': profession_counts.index,
        'occupation': [occupation_map.get(code, "Unknown") for code in profession_counts.index],
        'count': profession_counts.values,
        'percentage': profession_counts.values / len(data) * 100
    })
    
    print(f"Number of unique occupations: {len(df_counts)}")
    print("\nTop 5 most common occupations:")
    top_5 = df_counts.sort_values('count', ascending=False).head(5)
    for _, row in top_5.iterrows():
        print(f"  {row['occupation']}: {row['count']} samples ({row['percentage']:.2f}%)")
    
    print("\nBottom 5 least common occupations:")
    bottom_5 = df_counts.sort_values('count').head(5)
    for _, row in bottom_5.iterrows():
        print(f"  {row['occupation']}: {row['count']} samples ({row['percentage']:.2f}%)")
    
    # Plot occupation distribution
    plt.figure(figsize=(12, 8))
    sns.barplot(x='occupation', y='count', data=df_counts.sort_values('count', ascending=False))
    plt.xticks(rotation=90)
    plt.title('Number of Samples per Occupation')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/occupation_distribution.png')
    plt.close()
    
    return df_counts

def analyze_gender_distribution(data, occupation_map):
    """Analyze the gender distribution in the dataset."""
    print("\n=== Gender Distribution ===")
    
    # Overall gender distribution
    gender_counts = data['gender'].value_counts()
    total = len(data)
    print("Overall gender distribution:")
    for gender, count in gender_counts.items():
        gender_label = "Male" if gender == 0 else "Female"
        percentage = count / total * 100
        print(f"  {gender_label}: {count} samples ({percentage:.2f}%)")
    
    # Gender distribution by occupation
    gender_by_profession = data.groupby(['profession', 'gender']).size().unstack(fill_value=0)
    
    # Rename columns for clarity
    gender_by_profession.columns = ['Male', 'Female']
    
    # Calculate percentages
    gender_by_profession['Total'] = gender_by_profession['Male'] + gender_by_profession['Female']
    gender_by_profession['Male %'] = gender_by_profession['Male'] / gender_by_profession['Total'] * 100
    gender_by_profession['Female %'] = gender_by_profession['Female'] / gender_by_profession['Total'] * 100
    
    # Add occupation names
    gender_by_profession['Occupation'] = [occupation_map.get(code, "Unknown") for code in gender_by_profession.index]
    
    # Sort by female percentage
    gender_by_profession = gender_by_profession.sort_values('Female %', ascending=False)
    
    print("\nGender distribution by occupation:")
    for code, row in gender_by_profession.iterrows():
        occupation = occupation_map.get(code, "Unknown")
        print(f"  {occupation}: {row['Male']} males ({row['Male %']:.2f}%), {row['Female']} females ({row['Female %']:.2f}%)")
    
    # Plot gender distribution by occupation
    plt.figure(figsize=(14, 10))
    
    # Create a DataFrame for plotting
    plot_data = gender_by_profession.reset_index()
    plot_data = plot_data.sort_values('Female %', ascending=False)
    
    # Create the plot
    sns.set_color_codes("pastel")
    sns.barplot(x="Occupation", y="Male %", data=plot_data, label="Male", color="b")
    sns.set_color_codes("muted")
    sns.barplot(x="Occupation", y="Female %", data=plot_data, label="Female", color="r")
    
    # Add labels and title
    plt.title('Gender Distribution by Occupation')
    plt.xticks(rotation=90)
    plt.ylabel('Percentage')
    plt.legend(ncol=2, loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/gender_distribution_by_occupation.png')
    plt.close()
    
    return gender_by_profession

def analyze_text_statistics(data):
    """Analyze text length and other statistics."""
    print("\n=== Text Statistics ===")
    
    # Calculate text lengths
    data['text_length'] = data['hard_text'].str.len()
    data['word_count'] = data['hard_text'].str.split().str.len()
    
    # Overall statistics
    print("Text length statistics (characters):")
    print(f"  Mean: {data['text_length'].mean():.2f}")
    print(f"  Median: {data['text_length'].median():.2f}")
    print(f"  Min: {data['text_length'].min()}")
    print(f"  Max: {data['text_length'].max()}")
    print(f"  Standard deviation: {data['text_length'].std():.2f}")
    
    print("\nWord count statistics:")
    print(f"  Mean: {data['word_count'].mean():.2f}")
    print(f"  Median: {data['word_count'].median():.2f}")
    print(f"  Min: {data['word_count'].min()}")
    print(f"  Max: {data['word_count'].max()}")
    print(f"  Standard deviation: {data['word_count'].std():.2f}")
    
    # Text length distribution by occupation
    text_by_profession = data.groupby('profession')['text_length'].agg(['mean', 'median', 'min', 'max', 'std'])
    
    # Add occupation names
    occupation_map = dict(load_occupation_list())
    text_by_profession['occupation'] = [occupation_map.get(code, "Unknown") for code in text_by_profession.index]
    
    # Sort by mean text length
    text_by_profession = text_by_profession.sort_values('mean', ascending=False)
    
    print("\nText length statistics by occupation (top 5 longest):")
    for code, row in text_by_profession.head(5).iterrows():
        occupation = occupation_map.get(code, "Unknown")
        print(f"  {occupation}: mean={row['mean']:.2f}, median={row['median']:.2f}, min={row['min']}, max={row['max']}")
    
    # Plot text length distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data['text_length'], bins=50, kde=True)
    plt.title('Distribution of Text Lengths')
    plt.xlabel('Text Length (characters)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/text_length_distribution.png')
    plt.close()
    
    # Plot word count distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data['word_count'], bins=50, kde=True)
    plt.title('Distribution of Word Counts')
    plt.xlabel('Word Count')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/word_count_distribution.png')
    plt.close()
    
    return text_by_profession

def analyze_common_words(data, occupation_map, n_words=20):
    """Analyze most common words by occupation."""
    print("\n=== Common Words Analysis ===")
    
    # Function to get most common words
    def get_common_words(texts, n=n_words):
        words = ' '.join(texts).lower().split()
        # Filter out common stop words
        stop_words = {'the', 'and', 'a', 'to', 'of', 'in', 'is', 'he', 'she', 'has', 'with', 'at', 'as', 'for', 'on', 'his', 'her'}
        words = [word for word in words if word not in stop_words]
        return Counter(words).most_common(n)
    
    # Get common words by occupation
    common_words_by_occupation = {}
    for code in data['profession'].unique():
        occupation = occupation_map.get(code, "Unknown")
        texts = data[data['profession'] == code]['hard_text'].tolist()
        common_words = get_common_words(texts)
        common_words_by_occupation[occupation] = common_words
    
    # Print common words for a few occupations
    print("Most common words by occupation (top 3 occupations):")
    for i, (occupation, common_words) in enumerate(common_words_by_occupation.items()):
        if i >= 3:
            break
        print(f"\n  {occupation}:")
        for word, count in common_words[:10]:
            print(f"    {word}: {count}")
    
    return common_words_by_occupation

def main():
    """Main function to run the analysis."""
    # Load occupation list
    occupation_list = load_occupation_list()
    occupation_map = dict(occupation_list)
    print(f"Loaded {len(occupation_list)} occupations")
    
    # Load BiB data
    data = load_bib_data()
    
    # Analyze dataset structure
    analyze_dataset_structure(data)
    
    # Analyze occupation distribution
    occupation_counts = analyze_occupation_distribution(data, occupation_map)
    
    # Analyze gender distribution
    gender_distribution = analyze_gender_distribution(data, occupation_map)
    
    # Analyze text statistics
    text_statistics = analyze_text_statistics(data)
    
    # Analyze common words
    common_words = analyze_common_words(data, occupation_map)
    
    print("\nAnalysis complete. Results saved to the results directory.")

if __name__ == "__main__":
    main()
