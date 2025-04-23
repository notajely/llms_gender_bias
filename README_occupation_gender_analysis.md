# Occupation Gender Bias Analysis

This script analyzes gender bias in occupation embeddings using GPT-2. It calculates cosine similarities between occupation embeddings and gender term embeddings, and visualizes the results.

## Features

- Loads occupation data from CSV file
- Calculates embeddings for occupations and gender terms using GPT-2
- Computes cosine similarities between occupations and gender terms
- Creates visualizations:
  - Heatmap of similarities between occupations and gender terms
  - Barplot of gender projection scores
  - Scatter plot showing correlation between BLS gender ratios and model bias
- Saves results and visualizations to the results directory

## Requirements

- Python 3.6+
- PyTorch
- Transformers
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- tqdm

You can install the required packages using pip:

```bash
pip install torch transformers pandas numpy matplotlib seaborn scikit-learn tqdm
```

## Usage

1. Make sure your occupation data CSV file is in the correct location (default: `data/occupation_gender_data.csv`).
2. Run the script:

```bash
python occupation_gender_analysis.py
```

3. The script will:
   - Load the occupation data
   - Calculate embeddings and similarities
   - Create visualizations
   - Save results to the `./results` directory

## Configuration

You can modify the following parameters at the top of the script:

- `CSV_PATH`: Path to the occupation data CSV file
- `RESULTS_DIR`: Directory to save results and visualizations
- `MODEL_NAME`: Name of the GPT-2 model to use (e.g., 'gpt2', 'gpt2-medium')
- `GENDER_TERMS`: List of gender terms to use for similarity calculation
- `EMBEDDING_STRATEGY`: Strategy for calculating embeddings ('mean' or 'last')
- `FILTER_UNKNOWN`: Whether to filter out occupations with 'unknown' label

## Output

The script generates the following outputs in the results directory:

1. `occupation_gender_bias_results.csv`: CSV file containing similarity scores and bias values for each occupation
2. `occupation_gender_similarity_heatmap.png`: Heatmap visualization of similarities between occupations and gender terms
3. `occupation_gender_projection_barplot.png`: Barplot of gender projection scores for occupations
4. `occupation_gender_correlation.png`: Scatter plot showing correlation between BLS gender ratios and model bias

## Methodology

The script uses the following methodology:

1. **Embedding Calculation**:
   - For each occupation and gender term, calculate embeddings using GPT-2
   - For multi-word terms, use the mean of all token embeddings (or the last token embedding, depending on the strategy)

2. **Similarity Calculation**:
   - Calculate cosine similarity between each occupation embedding and each gender term embedding
   - Calculate male similarity as the average of similarities with 'he' and 'man'
   - Calculate female similarity as the average of similarities with 'she' and 'woman'
   - Calculate gender bias as male similarity minus female similarity

3. **Visualization**:
   - Create a heatmap of similarities between occupations and gender terms
   - Create a barplot of gender projection scores (bias values)
   - Create a scatter plot showing correlation between BLS gender ratios and model bias

## Notes

- The script uses GPU if available, otherwise falls back to CPU
- Processing all occupations may take some time, especially on CPU
- You can adjust the number of occupations shown in visualizations by modifying the `num_occupations` parameter in the visualization functions
