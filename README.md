# Gender Bias Analysis in Occupation Representations within GPT-2
**(Master Thesis Project)**

## Project Overview

This repository contains the code and analysis pipeline for a Master's thesis investigating occupational gender bias within the embedding space of the GPT-2 language model. The project examines how occupations are represented in relation to gender concepts using several analytical techniques:

1. **Cosine Similarity:** Assessing local semantic proximity between occupations and gendered terms ('he', 'she', 'man', 'woman').
2. **Gender Projection:** Measuring the alignment of occupation embeddings along a defined gender axis ('man' - 'woman').
3. **Semantic Clustering:** Exploring the organizational structure of occupation embeddings using KMeans clustering and PCA visualization based on contextual sentence embeddings.

The primary data sources include a curated list of 100 occupations with associated US Bureau of Labor Statistics (BLS) gender demographic data, and contextual sentences (derived from the Bias in Bios dataset) for a subset of 26 occupations used in the clustering analysis.

The analysis is structured as a series of Jupyter Notebooks for clear, reproducible steps.

## Directory Structure

```
├── notebooks/         # Jupyter notebooks for the main analysis pipeline (01-07).
├── data/
│   ├── raw/           # Original data files (e.g., BLS sheets, BiB pickles, raw occupation lists - *not tracked by Git*).
│   ├── processed/     # Intermediate processed data (e.g., sampled sentences pickle - *not tracked by Git*).
│   ├── occupation_gender_data.csv  # Primary input: 100 occupations w/ BLS data (Notebook 01 uses this).
│   └── bib_bias_cluster_analysis.csv # Input: Metadata for 28 (-> 26) occupations for clustering (Notebook 06 uses this).
├── results/           # Output directory for validated data, embeddings, scores, and plots.
│   ├── semantic_clustering/ # Subdirectory specifically for clustering outputs.
│   └── ...            # Other generated CSVs and PNGs.
├── src/               # (Optional) May contain helper Python scripts or alternative analyses.
├── logs/              # (Optional) Stores log files if generated.
├── .gitignore         # Specifies files/directories ignored by Git (e.g., data/, results/, logs/).
└── README.md          # This file.
```

**Note:** Specific data files required as input are detailed within the configuration sections of Notebooks `01` and `06`. Large data files in `data/` and generated files in `results/` are typically excluded from version control.

## Analysis Workflow (Notebook Pipeline)

The core analysis follows the sequence of Jupyter notebooks located in the `notebooks/` directory:

1. **`01_Data_Preparation_Occupation_Dictionary.ipynb`**: Loads and validates the primary 100-occupation dataset (`occupation_gender_data.csv`) containing BLS gender statistics and stereotype labels. Outputs a validated version to `results/`.
2. **`02_Embedding_Generation_GPT2.ipynb`**: Loads the GPT-2 model and generates *static* embeddings for the 100 occupations and the gender anchor terms using masked mean pooling. Saves embeddings to `results/`.
3. **`03_Analysis_Cosine_Similarity.ipynb`**: Calculates pairwise cosine similarities, derives the 'Similarity Bias' score, and visualizes results via a heatmap and a bias bar plot.
4. **`04_Analysis_Correlation_Similarity_vs_BLS.ipynb`**: Investigates the statistical correlation between 'Similarity Bias' and BLS demographic gender bias.
5. **`05_Analysis_Gender_Projection.ipynb`**: Defines the 'man'-'woman' gender axis, calculates geometric projection scores for occupations, and visualizes the distribution.
6. **`06_Data_Preparation_for_Clustering.ipynb`**: Prepares the dataset for clustering (26 occupations), loads contextual sentences, and generates *contextualized* GPT-2 embeddings.
7. **`07_Analysis_Semantic_Clustering.ipynb`**: Performs KMeans clustering and PCA, visualizes occupational semantic clusters, and saves outputs.

## Dependencies

* Python 3.10+
* Jupyter Lab / Jupyter Notebook
* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn
* tqdm
* torch
* transformers

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

