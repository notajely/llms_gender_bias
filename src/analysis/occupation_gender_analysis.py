#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Occupation Gender Bias Analysis Script

This script analyzes gender bias in occupation embeddings using GPT-2.
It calculates cosine similarities between occupation embeddings and gender term embeddings,
and visualizes the results.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import GPT2Tokenizer, GPT2Model
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Configuration
CSV_PATH = 'data/occupation_gender_data.csv'
RESULTS_DIR = './results'
MODEL_NAME = 'gpt2'
GENDER_TERMS = ['he', 'she', 'man', 'woman']
EMBEDDING_STRATEGY = 'mean'  # Options: 'mean', 'last'
FILTER_UNKNOWN = True  # Whether to filter out occupations with 'unknown' label

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set device (GPU if available, otherwise CPU)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

def load_data(csv_path):
    """Load occupation data from CSV file."""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Filter out occupations with unknown label if specified
    if FILTER_UNKNOWN:
        df = df[df['bls_label'] != 'unknown'].reset_index(drop=True)
        print(f"Filtered out occupations with 'unknown' label. {len(df)} occupations remaining.")
    
    return df

def clean_occupation_name(name):
    """Clean occupation name by removing special characters and replacing spaces with underscores."""
    if pd.isna(name):
        return ""
    # Remove quotes and commas
    cleaned = name.replace('"', '').replace(',', '')
    # Replace spaces with underscores for better tokenization
    cleaned = cleaned.replace(' ', '_')
    return cleaned

def load_model():
    """Load GPT-2 tokenizer and model."""
    print(f"Loading {MODEL_NAME} tokenizer and model...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2Model.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()  # Set model to evaluation mode
    
    # Add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    
    return tokenizer, model

def get_embedding(word_or_phrase, tokenizer, model, strategy=EMBEDDING_STRATEGY):
    """
    Get embedding for a word or phrase using the specified strategy.
    
    Args:
        word_or_phrase: The word or phrase to get embedding for
        tokenizer: The GPT-2 tokenizer
        model: The GPT-2 model
        strategy: The embedding strategy ('mean' or 'last')
        
    Returns:
        numpy array: The embedding vector
    """
    # Tokenize input
    inputs = tokenizer(word_or_phrase, return_tensors="pt").to(DEVICE)
    
    # Get model output
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get last hidden states
    last_hidden_states = outputs.last_hidden_state
    
    if strategy == 'mean':
        # Average all token embeddings
        embedding = last_hidden_states.mean(dim=1).squeeze().cpu().numpy()
    elif strategy == 'last':
        # Use last token embedding
        embedding = last_hidden_states[0, -1, :].cpu().numpy()
    else:
        raise ValueError(f"Unknown embedding strategy: {strategy}")
    
    return embedding

def calculate_similarities(occupations, tokenizer, model):
    """
    Calculate cosine similarities between occupations and gender terms.
    
    Args:
        occupations: List of occupation names
        tokenizer: The GPT-2 tokenizer
        model: The GPT-2 model
        
    Returns:
        DataFrame: Contains similarity scores for each occupation and gender term
    """
    print("Calculating embeddings for gender terms...")
    gender_embeddings = {term: get_embedding(term, tokenizer, model) for term in GENDER_TERMS}
    
    print(f"Calculating embeddings and similarities for {len(occupations)} occupations...")
    results = []
    
    for occupation in tqdm(occupations):
        # Skip empty occupations
        if not occupation or pd.isna(occupation):
            continue
            
        # Clean occupation name
        occupation_clean = clean_occupation_name(occupation)
        
        try:
            # Get occupation embedding
            occupation_emb = get_embedding(occupation_clean, tokenizer, model)
            
            # Calculate similarities with gender terms
            similarities = {}
            for gender_term, gender_emb in gender_embeddings.items():
                sim = cosine_similarity(
                    occupation_emb.reshape(1, -1), 
                    gender_emb.reshape(1, -1)
                )[0][0]
                similarities[gender_term] = sim
            
            # Calculate gender bias
            male_sim = np.mean([similarities['he'], similarities['man']])
            female_sim = np.mean([similarities['she'], similarities['woman']])
            bias = male_sim - female_sim
            
            # Add to results
            results.append({
                'occupation': occupation,
                'male_similarity': male_sim,
                'female_similarity': female_sim,
                'bias': bias,
                'he': similarities['he'],
                'she': similarities['she'],
                'man': similarities['man'],
                'woman': similarities['woman']
            })
        except Exception as e:
            print(f"Error processing occupation '{occupation}': {e}")
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    return df_results

def create_similarity_heatmap(df_results, num_occupations=30):
    """
    Create a heatmap of similarities between occupations and gender terms.
    
    Args:
        df_results: DataFrame with similarity results
        num_occupations: Number of occupations to include in the heatmap
        
    Returns:
        DataFrame: The data used for the heatmap
    """
    print("Creating similarity heatmap...")
    
    # Sort by absolute bias for more interesting visualization
    df_sorted = df_results.copy()
    df_sorted['abs_bias'] = df_sorted['bias'].abs()
    df_sorted = df_sorted.sort_values('abs_bias', ascending=False)
    
    # Select top N occupations
    df_selected = df_sorted.head(num_occupations)
    
    # Create heatmap data
    heatmap_data = df_selected[['occupation', 'he', 'she', 'man', 'woman']]
    heatmap_data = heatmap_data.set_index('occupation')
    
    # Create figure
    plt.figure(figsize=(10, 12))
    
    # Calculate min and max for color scaling
    vmin = heatmap_data.min().min() - 0.01
    vmax = heatmap_data.max().max() + 0.01
    
    # Create heatmap
    ax = sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        vmin=vmin, vmax=vmax,
        cbar_kws={'label': 'Cosine Similarity'},
        linewidths=0.4,
        linecolor='white',
        annot_kws={"size": 9}
    )
    
    plt.title('Cosine Similarity between Occupations and Gender Terms', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/occupation_gender_similarity_heatmap.png', dpi=300)
    plt.close()
    
    return heatmap_data

def create_gender_projection_barplot(df_results, df_occupation, num_occupations=30):
    """
    Create a barplot of gender projection scores for occupations.
    
    Args:
        df_results: DataFrame with similarity results
        df_occupation: DataFrame with occupation data
        num_occupations: Number of occupations to include in the barplot
        
    Returns:
        DataFrame: The data used for the barplot
    """
    print("Creating gender projection barplot...")
    
    # Merge results with occupation data to get BLS labels
    df_merged = pd.merge(
        df_results, 
        df_occupation[['occupation', 'bls_label']], 
        on='occupation', 
        how='left'
    )
    
    # Sort by bias
    df_sorted = df_merged.sort_values('bias', ascending=False)
    
    # Select top and bottom N/2 occupations
    top_n = df_sorted.head(num_occupations // 2)
    bottom_n = df_sorted.tail(num_occupations // 2)
    df_selected = pd.concat([top_n, bottom_n])
    
    # Create barplot data
    barplot_data = df_selected[['occupation', 'bias', 'bls_label']]
    barplot_data = barplot_data.rename(columns={'bias': 'Gender Projection'})
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Create barplot
    ax = sns.barplot(
        x='Gender Projection', 
        y='occupation', 
        data=barplot_data,
        palette='coolwarm', 
        orient='h', 
        hue='bls_label', 
        dodge=False
    )
    
    plt.title('Gender Projection Scores (Higher = More Male-Associated)', fontsize=16)
    plt.xlabel('Gender Projection Score', fontsize=14)
    plt.ylabel('Occupation', fontsize=14)
    plt.axvline(x=0, color='black', linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/occupation_gender_projection_barplot.png', dpi=300)
    plt.close()
    
    return barplot_data

def create_correlation_plot(df_results, df_occupation):
    """
    Create a scatter plot showing correlation between BLS gender ratios and model bias.
    
    Args:
        df_results: DataFrame with similarity results
        df_occupation: DataFrame with occupation data
        
    Returns:
        DataFrame: The data used for the scatter plot
    """
    print("Creating correlation plot...")
    
    # Merge results with occupation data
    df_merged = pd.merge(
        df_results, 
        df_occupation[['occupation', 'bls_female', 'bls_male', 'bls_label']], 
        on='occupation', 
        how='inner'
    )
    
    # Calculate BLS bias (male - female)
    df_merged['bls_bias'] = df_merged['bls_male'] - df_merged['bls_female']
    
    # Calculate correlation
    correlation = df_merged['bias'].corr(df_merged['bls_bias'])
    print(f"Correlation between GPT-2 gender bias and BLS gender ratio: {correlation:.4f}")
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Create scatter plot
    scatter = sns.scatterplot(
        x='bls_bias',
        y='bias',
        hue='bls_label',
        data=df_merged,
        palette='Set2',
        s=80,
        alpha=0.7
    )
    
    plt.title(f'Correlation between BLS Gender Ratio and GPT-2 Gender Bias (r = {correlation:.4f})', fontsize=16)
    plt.xlabel('BLS Gender Bias (Male % - Female %)', fontsize=14)
    plt.ylabel('GPT-2 Gender Bias', fontsize=14)
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/occupation_gender_correlation.png', dpi=300)
    plt.close()
    
    return df_merged

def main():
    """Main function to run the analysis."""
    # Load data
    df_occupation = load_data(CSV_PATH)
    
    # Load model
    tokenizer, model = load_model()
    
    # Calculate similarities
    df_results = calculate_similarities(df_occupation['occupation'].tolist(), tokenizer, model)
    
    # Save results
    df_results.to_csv(f'{RESULTS_DIR}/occupation_gender_bias_results.csv', index=False)
    print(f"Results saved to {RESULTS_DIR}/occupation_gender_bias_results.csv")
    
    # Create visualizations
    heatmap_data = create_similarity_heatmap(df_results)
    barplot_data = create_gender_projection_barplot(df_results, df_occupation)
    correlation_data = create_correlation_plot(df_results, df_occupation)
    
    print("Analysis complete. Results and visualizations saved to the results directory.")

if __name__ == "__main__":
    main()
