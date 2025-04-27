#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gender Vector Projection Visualization

This script implements the gender projection method described by Bolukbasi et al. (2016),
which measures the alignment of occupational embeddings along a learned gender direction
in vector space.

The gender axis is defined as the normalized vector difference between "man" and "woman",
and occupations are projected onto this axis to calculate their gender score.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import GPT2Tokenizer, GPT2Model

# Paths
RESULTS_DIR = './results'
OUTPUT_PATH = os.path.join(RESULTS_DIR, 'gender_vector_projection.png')
PROJECTION_DATA_PATH = os.path.join(RESULTS_DIR, 'gender_vector_projection_scores.csv')

# Model settings
MODEL_NAME = 'gpt2'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMBEDDING_STRATEGY = 'mean'  # 'mean' or 'last'

# Number of occupations to include in visualization
NUM_OCCUPATIONS = 40

def load_model():
    """Load GPT-2 tokenizer and model."""
    print(f"Loading {MODEL_NAME} tokenizer and model...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2Model.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()  # Set model to evaluation mode

    # Add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})  # Use EOS token as PAD
        model.resize_token_embeddings(len(tokenizer))
        print("Added EOS token as padding token.")

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
    inputs = tokenizer(
        word_or_phrase,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length
    ).to(DEVICE)

    # Get model output
    with torch.no_grad():
        outputs = model(**inputs)

    # Get last hidden states
    last_hidden_states = outputs.last_hidden_state

    if strategy == 'mean':
        # Apply attention mask for proper mean pooling
        mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden_states.size()).float()
        masked_embeddings = last_hidden_states * mask
        summed = torch.sum(masked_embeddings, 1)
        counts = torch.clamp(mask.sum(1), min=1e-9)
        embedding = (summed / counts).squeeze().cpu().numpy()
    elif strategy == 'last':
        # Find the index of the last actual token using the attention mask
        sequence_lengths = torch.sum(inputs['attention_mask'], dim=1)
        last_token_index = sequence_lengths[0] - 1
        embedding = last_hidden_states[0, last_token_index, :].cpu().numpy()
    else:
        raise ValueError(f"Unknown embedding strategy: {strategy}")

    return embedding

def clean_occupation_name(name):
    """Clean occupation name by removing special characters and replacing spaces with underscores."""
    if pd.isna(name):
        return ""
    # Remove quotes and commas
    cleaned = name.replace('"', '').replace(',', '')
    # Replace spaces with underscores for better tokenization
    cleaned = cleaned.replace(' ', '_')
    return cleaned

def calculate_gender_projections(occupations, tokenizer, model):
    """
    Calculate gender projections for occupations using the Bolukbasi et al. method.

    Args:
        occupations: List of occupation names
        tokenizer: The GPT-2 tokenizer
        model: The GPT-2 model

    Returns:
        DataFrame: Contains gender projection scores for each occupation
    """
    print("Calculating embeddings for gender terms...")

    # Get embeddings for 'man' and 'woman'
    man_embedding = get_embedding('man', tokenizer, model)
    woman_embedding = get_embedding('woman', tokenizer, model)

    # Calculate gender axis (man - woman)
    gender_axis = man_embedding - woman_embedding

    # Normalize gender axis
    gender_axis_norm = gender_axis / np.linalg.norm(gender_axis)

    print(f"Calculating gender projections for {len(occupations)} occupations...")
    results = []

    for occupation in occupations:
        # Skip empty occupations
        if not occupation or pd.isna(occupation):
            continue

        # Clean occupation name
        occupation_clean = clean_occupation_name(occupation)

        try:
            # Get occupation embedding
            occupation_emb = get_embedding(occupation_clean, tokenizer, model)

            # Calculate projection onto gender axis
            projection_score = np.dot(occupation_emb, gender_axis_norm)

            # Add to results
            results.append({
                'occupation': occupation,
                'projection_score': projection_score
            })
        except Exception as e:
            print(f"Error processing occupation '{occupation}': {e}")

    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    return df_results

def load_occupations(file_path=None):
    """
    Load occupation names from file or use default list.

    Args:
        file_path: Path to CSV file with occupation names

    Returns:
        list: List of occupation names
    """
    if file_path and os.path.exists(file_path):
        df = pd.read_csv(file_path)
        if 'occupation' in df.columns:
            return df['occupation'].tolist()

    # Default list of occupations if file not provided or invalid
    return [
        'nurse', 'doctor', 'engineer', 'teacher', 'programmer',
        'scientist', 'artist', 'writer', 'lawyer', 'accountant',
        'chef', 'pilot', 'architect', 'journalist', 'professor',
        'musician', 'designer', 'dentist', 'psychologist', 'librarian',
        'photographer', 'firefighter', 'police officer', 'actor', 'actress',
        'model', 'athlete', 'coach', 'manager', 'CEO',
        'secretary', 'assistant', 'receptionist', 'cashier', 'waiter',
        'waitress', 'cleaner', 'janitor', 'construction worker', 'mechanic'
    ]

def create_visualization(df, num_occupations=30, output_path=OUTPUT_PATH):
    """
    Create a horizontal bar chart visualization of gender projection scores.

    Args:
        df: DataFrame with occupation and projection_score columns
        num_occupations: Number of occupations to include
        output_path: Path to save the visualization
    """
    print("Creating gender projection visualization...")

    # Sort by projection score (descending)
    df_sorted = df.sort_values('projection_score', ascending=False)

    # Select a mix of high and low scoring occupations for better visualization
    # Take half from the top and half from the bottom
    top_half = df_sorted.head(num_occupations // 2)
    bottom_half = df_sorted.tail(num_occupations // 2)

    # Combine and sort again to show in descending order
    df_selected = pd.concat([top_half, bottom_half]).sort_values('projection_score', ascending=False).copy()

    # Create figure with appropriate height based on number of occupations
    fig_height = max(16, len(df_selected) * 0.5)  # Increase height for better readability
    plt.figure(figsize=(16, fig_height))  # Increase width for better x-axis display

    # Set the style
    sns.set_style("whitegrid")

    # Create a color map for the bars based on their values
    # Use a color gradient from blue (high values) to red (low values)

    # Get min and max values for normalization
    min_val = df_selected['projection_score'].min()
    max_val = df_selected['projection_score'].max()

    # Create a normalization function
    norm = mcolors.Normalize(min_val, max_val)

    # Create a colormap - blue for high values, red for low values
    # Use a colormap that's available in matplotlib
    cmap = plt.colormaps['coolwarm']

    # Generate colors for each bar based on its value
    colors = [cmap(norm(score)) for score in df_selected['projection_score']]

    # Create horizontal bar chart manually
    plt.barh(
        y=range(len(df_selected)),
        width=df_selected['projection_score'],  # Use raw scores directly
        color=colors,
        height=0.7
    )

    # Set y-tick labels to occupation names
    plt.yticks(range(len(df_selected)), df_selected['occupation'].tolist())

    # Set title and labels
    plt.title('Gender Projection Scores (Higher = More Male-Associated)', fontsize=16)
    plt.xlabel('Gender Projection Score', fontsize=14)
    plt.ylabel('Occupation', fontsize=14)

    # Add x-axis ticks and labels for better interpretation
    # Create evenly spaced ticks
    min_score = min(df_selected['projection_score'])
    max_score = max(df_selected['projection_score'])

    # Create more ticks for better readability
    step = (max_score - min_score) / 10  # 10 steps for more detail
    ticks = np.arange(round(min_score), round(max_score) + step, step)
    plt.xticks(ticks, fontsize=12)

    # Format tick labels to be more readable
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.1f}'))

    # Add a vertical line at the median for reference
    median_score = df_sorted['projection_score'].median()
    plt.axvline(x=median_score, color='black', linestyle='--', alpha=0.5, linewidth=1)
    plt.text(median_score, -1, f'Median: {median_score:.1f}', ha='center', va='top', fontsize=10)

    # Add gridlines for better readability
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Add a box around the plot
    plt.box(True)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")

    # Close figure
    plt.close()

def main():
    """Main function to run the analysis."""
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Path to existing occupation data
    existing_data_path = os.path.join(RESULTS_DIR, 'occupation_gender_bias_results.csv')

    # Check if we should use existing occupation data
    use_existing_occupations = os.path.exists(existing_data_path)

    # Check if projection data already exists
    if os.path.exists(PROJECTION_DATA_PATH):
        print(f"Loading existing projection data from {PROJECTION_DATA_PATH}...")
        df_projections = pd.read_csv(PROJECTION_DATA_PATH)
    else:
        # Load model
        tokenizer, model = load_model()

        # Load occupations
        if use_existing_occupations:
            print(f"Loading occupations from {existing_data_path}...")
            df_existing = pd.read_csv(existing_data_path)
            occupations = df_existing['occupation'].tolist()
        else:
            occupations = load_occupations()

        # Calculate gender projections
        df_projections = calculate_gender_projections(occupations, tokenizer, model)

        # Save projection data
        df_projections.to_csv(PROJECTION_DATA_PATH, index=False)
        print(f"Projection data saved to {PROJECTION_DATA_PATH}")

    # Create visualization
    create_visualization(df_projections, NUM_OCCUPATIONS, OUTPUT_PATH)

    print("Analysis complete.")

if __name__ == "__main__":
    main()
