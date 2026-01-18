import torch
from transformers import GPT2Tokenizer, GPT2Model
import numpy as np
from src.config import MODEL_NAME

def load_gpt2_model_and_tokenizer(device):
    """
    Loads GPT-2 model and tokenizer, sets up padding, and moves to device.
    """
    print(f"Loading {MODEL_NAME} tokenizer and model...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2Model.from_pretrained(MODEL_NAME)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model.to(device)
    model.eval()
    return tokenizer, model

def get_embedding(text, tokenizer, model, device):
    """
    Generates an embedding for the input text using masked mean pooling.
    """
    if not isinstance(text, str) or not text.strip():
        return None

    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_attention_mask=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        last_hidden_states = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
        
        masked_embeddings = last_hidden_states * mask_expanded
        summed_embeddings = torch.sum(masked_embeddings, 1)
        token_counts = torch.clamp(mask_expanded.sum(1), min=1e-9)
        mean_pooled_embedding = summed_embeddings / token_counts
        
        embedding_np = mean_pooled_embedding.squeeze().cpu().numpy()

        if np.isnan(embedding_np).any():
             return None

        return embedding_np

    except Exception as e:
        print(f"Error getting embedding for '{text}': {e}")
        return None

def get_contextual_embedding(occupation_name, sentences, tokenizer, model, device):
    """
    Generates a contextualized embedding for an occupation by averaging sentence embeddings.
    """
    sentence_embeddings = []
    if not sentences:
        return None

    for sentence in sentences:
        if not isinstance(sentence, str) or not sentence.strip():
            continue
        
        emb = get_embedding(sentence, tokenizer, model, device)
        if emb is not None:
            sentence_embeddings.append(emb)

    if sentence_embeddings:
        return np.mean(sentence_embeddings, axis=0)
    else:
        return None
