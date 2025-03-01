import pickle
import torch
from transformers import GPT2Tokenizer, GPT2Model
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import random
from tqdm import tqdm

# Define device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Load the BIB dataset
dataset_path = "bios_train.pkl"
with open(dataset_path, 'rb') as f:
    bios_data = pickle.load(f)

if isinstance(bios_data, list):
    bios_df = pd.DataFrame(bios_data)
else:
    bios_df = bios_data

# Extract profession words from BIB dataset classification labels
profession_list_from_readme = [
    "accountant", "architect", "attorney", "chiropractor", "comedian", "composer",
    "dentist", "dietitian", "dj", "filmmaker", "interior_designer", "journalist",
    "model", "nurse", "painter", "paralegal", "pastor", "personal_trainer",
    "photographer", "physician", "poet", "professor", "psychologist", "rapper",
    "software_engineer", "surgeon", "teacher", "yoga_teacher"
]
gender_terms = ["he", "she", "man", "woman",
                "person", "individual", "male", "female"]

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
model = model.to(device)

# get word embeddings


def get_word_embedding(word, tokenizer, model, device):
    inputs = tokenizer(word, return_tensors="pt").to(device)
    outputs = model(**inputs)
    word_embedding = outputs.last_hidden_state[:, 0, :].mean(dim=0)
    return word_embedding.detach().cpu().numpy()


# Obtain word embeddings for professions and gender terms
profession_embeddings = {
    profession: get_word_embedding(profession, tokenizer, model, device)
    for profession in profession_list_from_readme
}
gender_embeddings = {
    gender: get_word_embedding(gender, tokenizer, model, device)
    for gender in gender_terms
}

# Calculate cosine similarity between professions and gender terms
print("\nCosine Similarity between Professions and Gender Terms:")
similarity_scores = {}
for profession, profession_emb in profession_embeddings.items():
    similarity_scores[profession] = {}
    print(f"\nProfession: {profession}")
    for gender_term, gender_emb in gender_embeddings.items():
        similarity = cosine_similarity(
            np.array([profession_emb]), np.array([gender_emb]))[0][0]
        similarity_scores[profession][gender_term] = similarity
        print(f"  Cosine Similarity with '{gender_term}': {similarity:.4f}")

# PCA downscaling to 2D and visualization
profession_names = list(profession_embeddings.keys())
embeddings_for_pca = np.array([profession_embeddings[name]
                              for name in profession_names])
pca = PCA(n_components=2)
pca_result = pca.fit_transform(embeddings_for_pca)

plt.figure(figsize=(10, 8))
plt.scatter(pca_result[:, 0], pca_result[:, 1],
            c=range(len(profession_names)), cmap='viridis')
for i, profession in enumerate(profession_names):
    plt.annotate(profession, xy=(
        pca_result[i, 0], pca_result[i, 1]), fontsize=8)
plt.title('PCA Projection of Profession Word Embeddings')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Profession Index')
plt.grid(True)
plt.show()

# Define calculate_s and weat_score functions globally


def calculate_s(word, A, B, embeddings):
    emb_w = embeddings[word]
    mean_a = np.mean([embeddings[a] for a in A], axis=0)
    mean_b = np.mean([embeddings[b] for b in B], axis=0)
    return cosine_similarity(np.array([emb_w]), np.array([mean_a]))[0][0] - cosine_similarity(np.array([emb_w]), np.array([mean_b]))[0][0]


def weat_score(target_words_1, target_words_2, attribute_words_1, attribute_words_2, embeddings):
    S_diffs = [calculate_s(word, attribute_words_1, attribute_words_2, embeddings)
               for word in target_words_1 + target_words_2]
    mean_diff_targets = np.mean([calculate_s(word, attribute_words_1, attribute_words_2, embeddings)
                                 for word in target_words_1])
    std_dev_all_targets = np.std(S_diffs)
    if std_dev_all_targets == 0:
        return 0.0
    else:
        return (mean_diff_targets - np.mean([calculate_s(word, attribute_words_1, attribute_words_2, embeddings)
                                             for word in target_words_2])) / std_dev_all_targets


target_professions_male_stereotype = [
    "architect", "attorney", "dentist", "physician", "software_engineer",
    "surgeon", "composer", "filmmaker", "photographer", "rapper"
]
target_professions_female_stereotype = [
    "nurse", "teacher", "dietitian", "yoga_teacher", "interior_designer",
    "model", "paralegal", "personal_trainer"
]
attribute_male_terms = ["he", "man", "male"]
attribute_female_terms = ["she", "woman", "female"]

valid_target_professions_male = [
    p for p in target_professions_male_stereotype if p in profession_embeddings]
valid_target_professions_female = [
    p for p in target_professions_female_stereotype if p in profession_embeddings]

weat_embeddings = {}
for word in valid_target_professions_male + valid_target_professions_female + attribute_male_terms + attribute_female_terms:
    if word in profession_embeddings:
        weat_embeddings[word] = profession_embeddings[word]
    elif word in gender_embeddings:
        weat_embeddings[word] = gender_embeddings[word]
    else:
        print(f"Warning: {word} not found in any embedding dictionary.")

if (len(valid_target_professions_male) > 0 and len(valid_target_professions_female) > 0 and
        len(attribute_male_terms) > 0 and len(attribute_female_terms) > 0):
    effect_size = weat_score(valid_target_professions_male, valid_target_professions_female,
                             attribute_male_terms, attribute_female_terms, weat_embeddings)
    print(
        f"\nWEAT Test Effect Size (Male Stereotype Professions vs. Female Stereotype Professions, Attributes: Male vs. Female): {effect_size:.4f}")

    def permutation_test(target_words_1, target_words_2, attribute_words_1, attribute_words_2, embeddings, n_permutations=10000):
        observed_effect_size = weat_score(
            target_words_1, target_words_2, attribute_words_1, attribute_words_2, embeddings)
        combined_words = target_words_1 + target_words_2
        num_greater = 0
        for _ in tqdm(range(n_permutations), desc="Permutation Test"):
            shuffled = combined_words.copy()
            random.shuffle(shuffled)
            permuted_target_1 = shuffled[:len(target_words_1)]
            permuted_target_2 = shuffled[len(target_words_1):]
            permuted_effect_size = weat_score(
                permuted_target_1, permuted_target_2, attribute_words_1, attribute_words_2, embeddings)
            if permuted_effect_size >= observed_effect_size:
                num_greater += 1
        p_value = num_greater / n_permutations
        return p_value

    p_value = permutation_test(valid_target_professions_male, valid_target_professions_female,
                               attribute_male_terms, attribute_female_terms, weat_embeddings)
    print(f"P-value: {p_value:.4f}")

    print("\nIndividual word association scores:")
    for word in valid_target_professions_male + valid_target_professions_female:
        s = calculate_s(word, attribute_male_terms,
                        attribute_female_terms, weat_embeddings)
        print(f"  {word}: {s:.4f}")
else:
    print("\nWarning: Incomplete word sets for WEAT test. Please check that target professions and attribute words are included in the embedding dictionary.")
    print("Valid male stereotype professions:", valid_target_professions_male)
    print("Valid female stereotype professions:",
          valid_target_professions_female)

print("\nExperiment 1 code execution completed.")
