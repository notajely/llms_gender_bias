#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate contextualized embeddings for occupations using GPT-2.

This script:
1. Loads the sampled sentences for each occupation
2. Processes each sentence using GPT-2 to extract contextualized embeddings
3. Averages the embeddings for each occupation
4. Saves the embeddings to a file
"""

import os
import pickle
import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2Model
from tqdm import tqdm

# Configuration
INPUT_PATH = 'data/processed/bib_sampled_sentences.pkl'
OUTPUT_DIR = 'results/semantic_clustering'
EMBEDDING_DIM = 768  # GPT-2 embedding dimension

def load_sentences():
    """Load the sampled sentences."""
    print(f"Loading sentences from {INPUT_PATH}...")
    with open(INPUT_PATH, 'rb') as f:
        sentences = pickle.load(f)
    return sentences

def load_model():
    """Load the GPT-2 model and tokenizer."""
    print("Loading GPT-2 model and tokenizer...")
    # Set device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2').to(device)
    model.eval()  # Set model to evaluation mode
    
    return tokenizer, model, device

def get_sentence_embedding(sentence, tokenizer, model, device):
    """
    Get embedding for a sentence using GPT-2.
    
    Args:
        sentence: The sentence to get embedding for
        tokenizer: The GPT-2 tokenizer
        model: The GPT-2 model
        device: The device to use
        
    Returns:
        numpy array: The embedding vector
    """
    # Tokenize input
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    # Get model output
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get last hidden states
    last_hidden_states = outputs.last_hidden_state
    
    # Average all token embeddings (mean pooling)
    embedding = last_hidden_states.mean(dim=1).squeeze().cpu().numpy()
    
    return embedding

def generate_embeddings(sentences, tokenizer, model, device):
    """
    Generate embeddings for each occupation.
    
    Args:
        sentences: Dictionary mapping occupation names to lists of sentences
        tokenizer: The GPT-2 tokenizer
        model: The GPT-2 model
        device: The device to use
        
    Returns:
        Dictionary mapping occupation names to embedding vectors
    """
    occupation_embeddings = {}
    
    for occupation, occupation_sentences in tqdm(sentences.items(), desc="Processing occupations"):
        # Process each sentence
        sentence_embeddings = []
        for sentence in tqdm(occupation_sentences, desc=f"Processing {occupation}", leave=False):
            embedding = get_sentence_embedding(sentence, tokenizer, model, device)
            sentence_embeddings.append(embedding)
        
        # Average the embeddings
        occupation_embeddings[occupation] = np.mean(sentence_embeddings, axis=0)
    
    return occupation_embeddings

def save_embeddings(embeddings, output_dir):
    """
    Save the embeddings to a file.
    
    Args:
        embeddings: Dictionary mapping occupation names to embedding vectors
        output_dir: Directory to save the embeddings
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as pickle
    output_path = os.path.join(output_dir, 'occupation_embeddings.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)
    
    # Also save as numpy array for easier loading in other scripts
    occupation_names = list(embeddings.keys())
    embedding_matrix = np.array([embeddings[name] for name in occupation_names])
    
    np_output_path = os.path.join(output_dir, 'occupation_embeddings.npy')
    np.save(np_output_path, embedding_matrix)
    
    # Save occupation names
    names_output_path = os.path.join(output_dir, 'occupation_names.pkl')
    with open(names_output_path, 'wb') as f:
        pickle.dump(occupation_names, f)
    
    print(f"Saved embeddings for {len(embeddings)} occupations to {output_path}")
    print(f"Also saved as numpy array to {np_output_path}")
    print(f"Saved occupation names to {names_output_path}")

def main():
    """Main function to generate and save embeddings."""
    # Load sentences
    sentences = load_sentences()
    print(f"Loaded sentences for {len(sentences)} occupations")
    
    # Load model
    tokenizer, model, device = load_model()
    
    # Generate embeddings
    embeddings = generate_embeddings(sentences, tokenizer, model, device)
    
    # Save embeddings
    save_embeddings(embeddings, OUTPUT_DIR)

if __name__ == "__main__":
    main()
