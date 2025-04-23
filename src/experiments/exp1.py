#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Occupation Gender Bias Analysis Script

This script analyzes gender bias in occupation embeddings using GPT-2.
It calculates both:
1. Cosine similarities between occupation embeddings and gender term embeddings
2. Geometric gender projections by projecting occupation vectors onto the man-woman vector

The script then visualizes the results using heatmaps, bar plots, and scatter plots.
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
CSV_PATH = 'data/occupation_gender_data.csv' # Make sure this path is correct
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
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded {len(df)} rows.")
    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
        raise
    except Exception as e:
        print(f"Error loading CSV: {e}")
        raise

    # Basic cleaning of occupation names (remove extra quotes if any, trim whitespace)
    df['occupation'] = df['occupation'].str.strip().str.replace('^"|"$', '', regex=True)
    df = df.dropna(subset=['occupation']) # Remove rows where occupation is NaN after cleaning
    df = df[df['occupation'] != ''] # Remove rows with empty occupation names

    # Filter out occupations with unknown label if specified
    if FILTER_UNKNOWN:
        # Ensure 'bls_label' column exists
        if 'bls_label' in df.columns:
            original_count = len(df)
            df = df[df['bls_label'] != 'unknown'].reset_index(drop=True)
            print(f"Filtered out occupations with 'unknown' label. {original_count - len(df)} rows removed. {len(df)} occupations remaining.")
        else:
            print("Warning: 'bls_label' column not found. Cannot filter by 'unknown'.")

    # Drop duplicates based on the cleaned occupation name
    df = df.drop_duplicates(subset=['occupation'])
    print(f"Dropped duplicate occupations. {len(df)} unique occupations remaining.")

    return df

def clean_occupation_name(name):
    """Clean occupation name by removing specific special characters (quotes, commas)."""
    if pd.isna(name):
        return ""
    # Remove specific unwanted characters
    cleaned = name.replace('"', '').replace(',', '')
    # Trim leading/trailing whitespace
    cleaned = cleaned.strip()
    # Keep spaces, do NOT replace with underscores
    return cleaned

def load_model():
    """Load GPT-2 tokenizer and model."""
    print(f"Loading {MODEL_NAME} tokenizer and model...")
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
        model = GPT2Model.from_pretrained(MODEL_NAME).to(DEVICE)
        model.eval()  # Set model to evaluation mode
    except Exception as e:
        print(f"Error loading model {MODEL_NAME}: {e}")
        raise

    # Add padding token if missing (standard for GPT-2)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token}) # Use EOS token as PAD
        model.resize_token_embeddings(len(tokenizer))
        print("Added EOS token as padding token.")

    return tokenizer, model

def get_embedding(word_or_phrase, tokenizer, model, strategy=EMBEDDING_STRATEGY):
    """
    Get embedding for a word or phrase using the specified strategy, handling potential errors.

    Args:
        word_or_phrase (str): The word or phrase to get embedding for
        tokenizer: The GPT-2 tokenizer
        model: The GPT-2 model
        strategy (str): The embedding strategy ('mean' or 'last')

    Returns:
        numpy array or None: The embedding vector, or None if an error occurs
    """
    if not isinstance(word_or_phrase, str) or not word_or_phrase:
        print(f"Warning: Invalid input for embedding: {word_or_phrase}")
        return None

    try:
        # Tokenize input - Use padding and truncation
        inputs = tokenizer(
            word_or_phrase,
            return_tensors="pt",
            padding=True, # Pad to max length or batch length
            truncation=True, # Truncate if longer than model max length
            max_length=tokenizer.model_max_length
            ).to(DEVICE)

        # Get model output
        with torch.no_grad():
            outputs = model(**inputs)

        # Get last hidden states
        last_hidden_states = outputs.last_hidden_state # Shape: (batch_size, sequence_length, hidden_size)

        if strategy == 'mean':
            # --- CORRECTED Masked Mean Pooling ---
            mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden_states.size()).float()
            masked_embeddings = last_hidden_states * mask
            summed = torch.sum(masked_embeddings, 1)
            # Clamp prevents division by zero for empty sequences (though unlikely with tokenizer)
            counts = torch.clamp(mask.sum(1), min=1e-9)
            mean_pooled = summed / counts
            # Squeeze removes the batch dimension (assumes batch size 1)
            embedding = mean_pooled.squeeze().cpu().numpy()
            # --- End Correction ---

        elif strategy == 'last':
             # --- CORRECTED Last Token Strategy ---
            # Find the index of the last *actual* token using the attention mask
            sequence_lengths = torch.sum(inputs['attention_mask'], dim=1)
            # Index is length - 1 (adjusting for 0-based index)
            last_token_index = sequence_lengths[0] - 1
            # Handle potential empty input after tokenization (very unlikely)
            if last_token_index < 0:
                 print(f"Warning: No valid tokens found for '{word_or_phrase}' after tokenization.")
                 return None
            embedding = last_hidden_states[0, last_token_index, :].cpu().numpy()
             # --- End Correction ---
        else:
            raise ValueError(f"Unknown embedding strategy: {strategy}")

        # Check if embedding calculation resulted in NaNs (can happen in rare cases)
        if np.isnan(embedding).any():
            print(f"Warning: NaN detected in embedding for '{word_or_phrase}'. Skipping.")
            return None

        return embedding

    except Exception as e:
        print(f"Error getting embedding for '{word_or_phrase}': {e}")
        return None


def calculate_similarities_and_projections(occupations, tokenizer, model):
    """
    Calculate cosine similarities and geometric gender projections for occupations.

    Args:
        occupations (list): List of occupation names
        tokenizer: The GPT-2 tokenizer
        model: The GPT-2 model

    Returns:
        pd.DataFrame: Contains similarity scores and projection scores for each occupation
    """
    print("Calculating embeddings for gender terms...")
    gender_embeddings = {}
    for term in GENDER_TERMS:
        emb = get_embedding(term, tokenizer, model)
        if emb is None:
            print(f"FATAL: Could not get embedding for essential gender term '{term}'. Exiting.")
            raise ValueError(f"Failed to embed gender term: {term}")
        gender_embeddings[term] = emb

    # Calculate gender axis for geometric projection (man - woman)
    try:
        gender_axis = gender_embeddings['man'] - gender_embeddings['woman']
        # Normalize gender axis - handle potential zero vector
        norm = np.linalg.norm(gender_axis)
        if norm == 0:
             print("FATAL: Gender axis vector has zero magnitude ('man' and 'woman' embeddings are identical?).")
             raise ValueError("Zero magnitude gender axis")
        gender_axis_norm = gender_axis / norm
        print("Calculated normalized gender axis.")
    except Exception as e:
        print(f"FATAL: Error calculating gender axis: {e}")
        raise

    print(f"Calculating embeddings, similarities, and projections for {len(occupations)} occupations...")
    results = []

    for occupation in tqdm(occupations):
        # Clean occupation name FIRST
        occupation_clean = clean_occupation_name(occupation)

        # Skip empty cleaned occupations
        if not occupation_clean:
            print(f"Skipping empty occupation entry (original: '{occupation}').")
            continue

        occupation_emb = get_embedding(occupation_clean, tokenizer, model)

        # Skip if embedding failed
        if occupation_emb is None:
            print(f"Skipping occupation '{occupation}' due to embedding error.")
            continue

        try:
            # Calculate similarities with gender terms
            similarities = {}
            for gender_term, gender_emb in gender_embeddings.items():
                # Ensure embeddings are 2D for cosine_similarity
                occ_emb_2d = occupation_emb.reshape(1, -1)
                gen_emb_2d = gender_emb.reshape(1, -1)
                sim = cosine_similarity(occ_emb_2d, gen_emb_2d)[0][0]
                similarities[gender_term] = sim

            # Calculate similarity-based gender bias (for comparison)
            male_sim = np.mean([similarities['he'], similarities['man']])
            female_sim = np.mean([similarities['she'], similarities['woman']])
            sim_bias = male_sim - female_sim

            # Calculate geometric gender projection
            # Project occupation vector onto the normalized gender axis
            projection_score = np.dot(occupation_emb, gender_axis_norm)

            # Add to results
            results.append({
                'occupation': occupation, # Store original name for readability
                'male_similarity': male_sim,
                'female_similarity': female_sim,
                'similarity_bias': sim_bias,  # Similarity-based bias
                'projection_score': projection_score,  # Geometric projection score
                'he': similarities['he'],
                'she': similarities['she'],
                'man': similarities['man'],
                'woman': similarities['woman']
            })
        except Exception as e:
            # Catch errors during calculation for a specific occupation
            print(f"Error calculating metrics for occupation '{occupation}': {e}")

    # Convert to DataFrame
    if not results:
        print("Warning: No results were generated.")
        return pd.DataFrame()

    df_results = pd.DataFrame(results)
    return df_results

def create_similarity_heatmap(df_results, num_occupations=30):
    """
    Create a heatmap of similarities between occupations and gender terms.
    Sorts by absolute projection score to show most polarized occupations.
    """
    print("Creating similarity heatmap...")
    if df_results.empty or 'projection_score' not in df_results.columns:
        print("Skipping heatmap: No results data or projection scores available.")
        return None

    # Sort by absolute projection score for more interesting visualization
    df_sorted = df_results.copy()
    df_sorted['abs_projection'] = df_sorted['projection_score'].abs()
    df_sorted = df_sorted.sort_values('abs_projection', ascending=False)

    # Select top N occupations if dataframe has enough rows
    num_to_select = min(num_occupations, len(df_sorted))
    if num_to_select == 0:
        print("Skipping heatmap: No occupations to display.")
        return None
    df_selected = df_sorted.head(num_to_select)

    # Create heatmap data
    heatmap_data = df_selected[['occupation', 'he', 'she', 'man', 'woman']]
    heatmap_data = heatmap_data.set_index('occupation')

    # Adjust figure height based on number of occupations
    fig_height = max(8, num_to_select * 0.35)
    plt.figure(figsize=(10, fig_height))

    # Calculate min and max for consistent color scaling, handle potential NaNs
    valid_data = heatmap_data.dropna().values # Use only non-NaN values for scaling
    if valid_data.size == 0:
         print("Warning: No valid data for heatmap color scaling.")
         vmin, vmax = 0.8, 1.0 # Default fallback scaling
    else:
         vmin = np.nanmin(valid_data) - 0.01
         vmax = np.nanmax(valid_data) + 0.01

    # Create heatmap
    try:
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
        plt.yticks(rotation=0) # Ensure y-axis labels are horizontal
        plt.tight_layout()
        save_path = f'{RESULTS_DIR}/occupation_gender_similarity_heatmap.png'
        plt.savefig(save_path, dpi=300)
        print(f"Heatmap saved to {save_path}")
        plt.close()
        return heatmap_data
    except Exception as e:
        print(f"Error creating heatmap: {e}")
        plt.close()
        return None

def create_geometric_projection_barplot(df_results, df_occupation, num_occupations=30):
    """
    Create a barplot of geometric gender projection scores for occupations.
    Shows top N/2 most male-projected and bottom N/2 most female-projected.
    """
    print("Creating geometric gender projection barplot...")
    if df_results.empty or 'projection_score' not in df_results.columns:
        print("Skipping geometric projection barplot: No results data or projection scores.")
        return None

    # Merge results with occupation data to get BLS labels if available
    if 'bls_label' in df_occupation.columns:
        df_merged = pd.merge(
            df_results,
            df_occupation[['occupation', 'bls_label']],
            on='occupation',
            how='left'
        )
        hue_col = 'bls_label'
        palette = 'Set2' # Use a palette suitable for categories
    else:
        print("Warning: 'bls_label' not found in occupation data. Barplot will not be colored by label.")
        df_merged = df_results.copy()
        hue_col = None
        palette = 'coolwarm' # Fallback palette

    # Sort by projection score
    df_sorted = df_merged.sort_values('projection_score', ascending=False).dropna(subset=['projection_score'])

    # Select top and bottom N/2 occupations if dataframe has enough rows
    n_each = num_occupations // 2
    if len(df_sorted) < num_occupations:
        print(f"Warning: Fewer than {num_occupations} occupations with valid scores. Plotting all {len(df_sorted)}.")
        df_selected = df_sorted
    elif len(df_sorted) >= n_each * 2:
        top_n = df_sorted.head(n_each)
        bottom_n = df_sorted.tail(n_each)
        df_selected = pd.concat([top_n, bottom_n]).sort_values('projection_score', ascending=False)
    else: # Handle cases with < n_each*2 but >= n_each
         print(f"Warning: Only {len(df_sorted)} occupations available for bar plot.")
         df_selected = df_sorted


    if df_selected.empty:
        print("Skipping geometric projection barplot: No occupations to display after sorting/filtering.")
        return None

    # Create figure
    fig_height = max(8, len(df_selected) * 0.35)
    plt.figure(figsize=(12, fig_height))

    # Create barplot
    try:
        ax = sns.barplot(
            x='projection_score', # Use projection_score here
            y='occupation',
            data=df_selected,
            palette=palette,
            orient='h',
            hue=hue_col, # Use BLS label for color if available
            dodge=False # Avoid dodging bars when hue is used
        )
        plt.title('Geometric Gender Projection Scores (Higher = More Male-Associated)', fontsize=16)
        plt.xlabel('Geometric Gender Projection Score', fontsize=14)
        plt.ylabel('Occupation', fontsize=14)
        plt.axvline(x=0, color='black', linestyle='--') # Line at zero projection
        if hue_col:
            plt.legend(title='BLS Label', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1] if hue_col else None) # Adjust layout if legend is present
        save_path = f'{RESULTS_DIR}/occupation_geometric_gender_projection_barplot.png'
        plt.savefig(save_path, dpi=300)
        print(f"Geometric projection barplot saved to {save_path}")
        plt.close()
        return df_selected[['occupation', 'projection_score', 'bls_label']] if hue_col else df_selected[['occupation', 'projection_score']]
    except Exception as e:
        print(f"Error creating geometric projection barplot: {e}")
        plt.close()
        return None


from scipy import stats
import matplotlib.pyplot as plt # Ensure matplotlib.pyplot is imported as plt
import seaborn as sns # Ensure seaborn is imported as sns
import pandas as pd # Ensure pandas is imported as pd
import numpy as np # Ensure numpy is imported as np


# REPLACE the existing create_correlation_plot function in your main script with this:
def create_correlation_plot(df_results, df_occupation):
    """
    Create a scatter plot visualizing the correlation between BLS female ratio
    and geometric gender projection scores, WITHOUT point annotations,
    and including correlation stats. Also generates a comparison plot for similarity bias.

    Args:
        df_results (pd.DataFrame): DataFrame with projection scores and similarity bias.
        df_occupation (pd.DataFrame): DataFrame with BLS occupation data.

    Returns:
        pd.DataFrame or None: The data used for the primary scatter plot, or None if creation fails.
    """
    print("Creating correlation plots (Female Ratio vs Projection / Similarity Bias)...")

    # --- Data Preparation ---
    required_results_cols = ['occupation', 'projection_score', 'similarity_bias']
    required_bls_cols = ['occupation', 'bls_female', 'bls_male', 'bls_label'] # Need bls_label for color

    if df_results.empty or not all(col in df_results.columns for col in required_results_cols):
        print("Skipping correlation plots: Missing required columns in results data.")
        return None
    if df_occupation.empty or not all(col in df_occupation.columns for col in required_bls_cols):
        print("Skipping correlation plots: Missing required columns in occupation data.")
        return None

    # Merge results with occupation data
    df_merged = pd.merge(
        df_results,
        df_occupation[required_bls_cols],
        on='occupation',
        how='inner' # Ensure only overlapping occupations with BLS data are included
    )

    # Clean up stereotype labels
    df_merged['bls_label'] = df_merged['bls_label'].astype(str).str.replace(' (proxy)', '', regex=False)

    # Filter to keep only the three main categories and drop NaNs
    categories = ['male-stereotyped', 'neutral', 'female-stereotyped']
    df_filtered = df_merged[df_merged['bls_label'].isin(categories)].copy()
    df_filtered = df_filtered.dropna(subset=['bls_female', 'bls_male', 'projection_score', 'similarity_bias'])

    if df_filtered.empty:
        print("Skipping correlation plots: No valid data after merging, cleaning, and filtering.")
        return None

    print(f"Plotting correlations for {len(df_filtered)} occupations.")
    df_filtered['bls_bias'] = df_filtered['bls_male'] - df_filtered['bls_female'] # Calculate BLS bias for comparison plot

    # --- Correlation Calculation ---
    pearson_r_proj, pearson_p_proj = np.nan, np.nan
    spearman_rho_proj, spearman_p_proj = np.nan, np.nan
    correlation_text_proj = "Correlation (Projection) could not be calculated."

    if len(df_filtered) > 1 and df_filtered['bls_female'].var() > 0 and df_filtered['projection_score'].var() > 0:
        try:
            pearson_r_proj, pearson_p_proj = stats.pearsonr(df_filtered['bls_female'].values, df_filtered['projection_score'].values)
            spearman_rho_proj, spearman_p_proj = stats.spearmanr(df_filtered['bls_female'].values, df_filtered['projection_score'].values)
            print(f"Pearson (Female Ratio vs Projection): r={pearson_r_proj:.4f}, p={pearson_p_proj:.4f}")
            print(f"Spearman (Female Ratio vs Projection): rho={spearman_rho_proj:.4f}, p={spearman_p_proj:.4f}")
            correlation_text_proj = (
                f'Pearson: r={pearson_r_proj:.3f} (p={pearson_p_proj:.3f})\n'
                f'Spearman: ρ={spearman_rho_proj:.3f} (p={spearman_p_proj:.3f})'
            )
        except Exception as e: print(f"Error calculating projection correlation: {e}")

    pearson_r_sim, pearson_p_sim = np.nan, np.nan
    spearman_rho_sim, spearman_p_sim = np.nan, np.nan
    correlation_text_sim = "Correlation (Similarity Bias) could not be calculated."

    if len(df_filtered) > 1 and df_filtered['bls_bias'].var() > 0 and df_filtered['similarity_bias'].var() > 0:
         try:
            pearson_r_sim, pearson_p_sim = stats.pearsonr(df_filtered['bls_bias'].values, df_filtered['similarity_bias'].values)
            spearman_rho_sim, spearman_p_sim = stats.spearmanr(df_filtered['bls_bias'].values, df_filtered['similarity_bias'].values)
            print(f"Pearson (BLS Bias vs Similarity Bias): r={pearson_r_sim:.4f}, p={pearson_p_sim:.4f}")
            print(f"Spearman (BLS Bias vs Similarity Bias): rho={spearman_rho_sim:.4f}, p={spearman_p_sim:.4f}")
            correlation_text_sim = (
                 f'Pearson: r={pearson_r_sim:.3f} (p={pearson_p_sim:.3f})\n'
                 f'Spearman: ρ={spearman_rho_sim:.3f} (p={spearman_p_sim:.3f})'
             )
         except Exception as e: print(f"Error calculating similarity bias correlation: {e}")


    # --- Plotting ---
    plot_created = False
    try:
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 10)) # Keep a decent size

        # Define specific colors
        category_colors = {
            'male-stereotyped': '#95B3D7',
            'neutral': '#9DCDA9',
            'female-stereotyped': '#FFB598'
        }

        ax = sns.scatterplot(
            x='bls_female', # X-axis is Female Ratio
            y='projection_score', # Y-axis is Geometric Projection Score
            hue='bls_label',
            hue_order=categories,
            palette=category_colors,
            s=80,  # Adjusted point size slightly
            alpha=0.8,
            data=df_filtered,
            edgecolor="w",
            linewidth=0.5
        )

        # --- Annotation loop REMOVED ---
        # # Add occupation labels to points (REMOVED as requested)
        # for i, row in df_filtered.iterrows():
        #     plt.annotate(...)

        # Add regression line
        sns.regplot(
            x='bls_female',
            y='projection_score',
            data=df_filtered,
            scatter=False,
            ci=95,
            line_kws={'color': 'black', 'linestyle': '--', 'linewidth': 1.5}
        )

        # Add correlation text box
        plt.text(
            0.03, 0.97,
            correlation_text_proj, # Use the calculated text
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85)
        )

        # Set labels and title
        ax.set_title('Female Ratio vs. Geometric Projection Score', fontsize=16)
        ax.set_xlabel('Female Ratio in Occupation (BLS Data)', fontsize=14)
        ax.set_ylabel('Geometric Gender Projection Score', fontsize=14)

        # Set reasonable axis limits
        ax.set_xlim(-0.05, 1.05)
        y_min, y_max = df_filtered['projection_score'].min(), df_filtered['projection_score'].max()
        y_pad = max((y_max - y_min) * 0.1, 1) # Add padding, ensure it's at least 1 unit
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

        # Add reference line at y=0
        plt.axhline(y=0, color='grey', linestyle='--', linewidth=1, alpha=0.7)

        # Adjust legend
        plt.legend(title='BLS Label', fontsize=11, title_fontsize=12, loc='upper left', bbox_to_anchor=(1.02, 1))

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 0.85, 1])

        # Save the figure
        save_path_png = f'{RESULTS_DIR}/correlation_analysis_femaleratio_vs_projection.png'
        save_path_pdf = f'{RESULTS_DIR}/correlation_analysis_femaleratio_vs_projection.pdf'
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(save_path_pdf, bbox_inches='tight')
        print(f"Correlation plot (Female Ratio vs Projection) saved to {save_path_png}")
        plt.close()
        plot_created = True # Mark as successful

    except Exception as e:
        print(f"Error creating correlation plot (Female Ratio vs Projection): {e}")
        plt.close()


    # --- Plotting: Similarity Bias vs BLS Bias (Optional Comparison Plot) ---
    # This plot uses BLS Bias on X-axis as in the original example figures
    # If you ONLY want the Female Ratio plot, you can remove this section
    try:
        plt.figure(figsize=(12, 10))
        ax2 = sns.scatterplot(
            x='bls_bias', # X-axis is BLS Bias (Male % - Female %)
            y='similarity_bias', # Y-axis is Similarity Bias Score
            hue='bls_label',
            hue_order=categories,
            palette=category_colors,
            s=80,
            alpha=0.8,
            data=df_filtered,
            edgecolor="w",
            linewidth=0.5
        )

        # --- Annotation loop REMOVED ---

        # Add regression line
        sns.regplot(
            x='bls_bias',
            y='similarity_bias',
            data=df_filtered,
            scatter=False,
            ci=95,
            line_kws={'color': 'black', 'linestyle': '--', 'linewidth': 1.5}
        )

         # Add correlation text box
        plt.text(
            0.03, 0.97,
            correlation_text_sim, # Use the calculated text
            transform=ax2.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85)
        )

        ax2.set_title('Similarity-Based Gender Bias vs BLS Gender Bias', fontsize=16)
        ax2.set_xlabel('BLS Gender Bias (Male % - Female %)', fontsize=14)
        ax2.set_ylabel('Similarity-Based Gender Bias Score', fontsize=14)
        ax2.set_xlim(-1.05, 1.05) # BLS bias ranges from -1 to 1
        sim_y_min, sim_y_max = df_filtered['similarity_bias'].min(), df_filtered['similarity_bias'].max()
        sim_y_pad = max((sim_y_max - sim_y_min) * 0.1, 0.001) # Add padding
        ax2.set_ylim(sim_y_min - sim_y_pad, sim_y_max + sim_y_pad)
        plt.axhline(y=0, color='grey', linestyle='--', linewidth=1, alpha=0.7)
        plt.axvline(x=0, color='grey', linestyle='--', linewidth=1, alpha=0.7) # Add x=0 line for BLS Bias
        plt.legend(title='BLS Label', fontsize=11, title_fontsize=12, loc='upper left', bbox_to_anchor=(1.02, 1))
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        save_path_sim_png = f'{RESULTS_DIR}/correlation_analysis_blsbias_vs_simbias.png'
        save_path_sim_pdf = f'{RESULTS_DIR}/correlation_analysis_blsbias_vs_simbias.pdf'
        plt.savefig(save_path_sim_png, dpi=300, bbox_inches='tight')
        plt.savefig(save_path_sim_pdf, bbox_inches='tight')
        print(f"Correlation plot (BLS Bias vs Similarity Bias) saved to {save_path_sim_png}")
        plt.close()

    except Exception as e:
         print(f"Error creating correlation plot (BLS Bias vs Similarity Bias): {e}")
         plt.close()


    # Return the data used for the primary plot (Female Ratio vs Projection)
    return df_filtered if plot_created else None


def create_similarity_bias_barplot(df_results, df_occupation, num_occupations=30):
    """
    Create a barplot of similarity-based gender bias scores for comparison.
    Shows top N/2 most male-biased and bottom N/2 most female-biased based on similarity.
    """
    print("Creating similarity-based gender bias barplot...")
    if df_results.empty or 'similarity_bias' not in df_results.columns:
        print("Skipping similarity bias barplot: No results data or similarity bias scores.")
        return None

    # Merge results with occupation data to get BLS labels if available
    if 'bls_label' in df_occupation.columns:
        df_merged = pd.merge(
            df_results,
            df_occupation[['occupation', 'bls_label']],
            on='occupation',
            how='left'
        )
        hue_col = 'bls_label'
        palette = 'Set2'
    else:
        print("Warning: 'bls_label' not found in occupation data. Similarity bias barplot will not be colored by label.")
        df_merged = df_results.copy()
        hue_col = None
        palette = 'coolwarm' # Fallback

    # Sort by similarity bias
    df_sorted = df_merged.sort_values('similarity_bias', ascending=False).dropna(subset=['similarity_bias'])

    # Select top and bottom N/2 occupations
    n_each = num_occupations // 2
    if len(df_sorted) < num_occupations:
        print(f"Warning: Fewer than {num_occupations} occupations with valid scores. Plotting all {len(df_sorted)}.")
        df_selected = df_sorted
    elif len(df_sorted) >= n_each * 2:
        top_n = df_sorted.head(n_each)
        bottom_n = df_sorted.tail(n_each)
        df_selected = pd.concat([top_n, bottom_n]).sort_values('similarity_bias', ascending=False)
    else:
         print(f"Warning: Only {len(df_sorted)} occupations available for similarity bar plot.")
         df_selected = df_sorted

    if df_selected.empty:
        print("Skipping similarity bias barplot: No occupations to display.")
        return None

    # Create figure
    fig_height = max(8, len(df_selected) * 0.35)
    plt.figure(figsize=(12, fig_height))

    # Create barplot
    try:
        ax = sns.barplot(
            x='similarity_bias', # Use similarity_bias here
            y='occupation',
            data=df_selected,
            palette=palette,
            orient='h',
            hue=hue_col,
            dodge=False
        )
        plt.title('Similarity-Based Gender Bias Scores (Higher = More Male-Associated)', fontsize=16)
        plt.xlabel('Similarity-Based Gender Bias Score', fontsize=14)
        plt.ylabel('Occupation', fontsize=14)
        plt.axvline(x=0, color='black', linestyle='--')
        if hue_col:
            plt.legend(title='BLS Label', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1] if hue_col else None) # Adjust layout if legend is present
        save_path = f'{RESULTS_DIR}/occupation_similarity_bias_barplot.png'
        plt.savefig(save_path, dpi=300)
        print(f"Similarity bias barplot saved to {save_path}")
        plt.close()
        return df_selected[['occupation', 'similarity_bias', 'bls_label']] if hue_col else df_selected[['occupation', 'similarity_bias']]
    except Exception as e:
        print(f"Error creating similarity bias barplot: {e}")
        plt.close()
        return None

def main():
    """Main function to run the analysis."""
    # Load data
    try:
        df_occupation = load_data(CSV_PATH)
        if df_occupation.empty:
            print("Loaded data is empty after filtering. Exiting.")
            return
    except FileNotFoundError:
        # Error message already printed in load_data
        return
    except Exception as e:
        # Error message already printed in load_data
        return

    # Load model
    try:
        tokenizer, model = load_model()
    except Exception as e:
        # Error message already printed in load_model
        return

    # Calculate similarities and projections
    try:
        df_results = calculate_similarities_and_projections(df_occupation['occupation'].tolist(), tokenizer, model)
        if df_results.empty:
            print("No results generated from calculations. Check input data and model.")
            return
    except Exception as e:
        print(f"FATAL Error during calculations: {e}")
        return

    # Save results
    results_path = f'{RESULTS_DIR}/occupation_gender_analysis_results.csv'
    df_results.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")

    # Create visualizations (functions include internal checks for empty data)
    create_similarity_heatmap(df_results)
    create_geometric_projection_barplot(df_results, df_occupation)
    create_similarity_bias_barplot(df_results, df_occupation) # Add call for comparison plot
    create_correlation_plot(df_results, df_occupation)

    print("\nAnalysis complete. Results and visualizations saved to the results directory.")

if __name__ == "__main__":
    main()