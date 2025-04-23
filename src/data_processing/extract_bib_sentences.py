#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract sentences from the Bias in Bios dataset for semantic clustering analysis.

This script:
1. Loads the Bias in Bios dataset
2. Extracts a fixed number of sentences for each occupation
3. Saves the extracted sentences to a file
"""

import os
import pickle
import pandas as pd
import random
import csv
from tqdm import tqdm

# Configuration
RAW_DATA_PATH = 'data/raw/bios_train.pkl'
OCCUPATION_LIST_PATH = 'data/bias_in_bia_occupation_list.csv'
OUTPUT_DIR = 'data/processed'
OUTPUT_FILE = 'bib_sampled_sentences.txt'
SAMPLES_PER_OCCUPATION = 20
RANDOM_SEED = 42

def load_occupation_list():
    """Load the occupation list from CSV file."""
    occupation_list = []
    with open(OCCUPATION_LIST_PATH, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            occupation_list.append((int(row[0]), row[1]))
    return occupation_list

def load_bib_data():
    """Load the Bias in Bios dataset."""
    print(f"Loading BiB data from {RAW_DATA_PATH}...")
    with open(RAW_DATA_PATH, 'rb') as f:
        data = pickle.load(f)
    return data

def extract_sentences(data, occupation_list, samples_per_occupation):
    """
    Extract sentences for each occupation.
    
    Args:
        data: BiB dataset
        occupation_list: List of (id, name) tuples for occupations
        samples_per_occupation: Number of samples to extract per occupation
        
    Returns:
        Dictionary mapping occupation names to lists of sentences
    """
    # Create a mapping from profession code to occupation name
    profession_to_occupation = {code: name for code, name in occupation_list}
    
    # Group data by profession
    profession_groups = {}
    for _, row in tqdm(data.iterrows(), total=len(data), desc="Grouping data by profession"):
        prof_code = row['profession']
        if prof_code in profession_to_occupation:
            if prof_code not in profession_groups:
                profession_groups[prof_code] = []
            profession_groups[prof_code].append(row['hard_text'])
    
    # Sample sentences for each occupation
    sampled_sentences = {}
    for prof_code, sentences in tqdm(profession_groups.items(), desc="Sampling sentences"):
        occupation = profession_to_occupation[prof_code]
        # If we have fewer sentences than requested, use all of them
        if len(sentences) <= samples_per_occupation:
            sampled_sentences[occupation] = sentences
        else:
            # Otherwise, sample randomly
            random.seed(RANDOM_SEED)
            sampled_sentences[occupation] = random.sample(sentences, samples_per_occupation)
    
    return sampled_sentences

def save_sentences(sentences, output_path):
    """
    Save extracted sentences to a file.
    
    Args:
        sentences: Dictionary mapping occupation names to lists of sentences
        output_path: Path to save the sentences
    """
    with open(output_path, 'w') as f:
        for occupation, occupation_sentences in sentences.items():
            for sentence in occupation_sentences:
                # Format: occupation|sentence
                f.write(f"{occupation}|{sentence}\n")
    
    # Also save as pickle for easier loading
    pickle_path = output_path.replace('.txt', '.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(sentences, f)
    
    print(f"Saved {sum(len(s) for s in sentences.values())} sentences to {output_path}")
    print(f"Also saved as pickle to {pickle_path}")

def main():
    """Main function to extract and save sentences."""
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load occupation list
    occupation_list = load_occupation_list()
    print(f"Loaded {len(occupation_list)} occupations")
    
    # Load BiB data
    data = load_bib_data()
    print(f"Loaded BiB data with {len(data)} entries")
    
    # Extract sentences
    sentences = extract_sentences(data, occupation_list, SAMPLES_PER_OCCUPATION)
    
    # Print statistics
    print("\nSampling statistics:")
    for occupation, occupation_sentences in sentences.items():
        print(f"{occupation}: {len(occupation_sentences)} sentences")
    
    # Save sentences
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    save_sentences(sentences, output_path)

if __name__ == "__main__":
    main()
