# Gender Bias Analysis in GPT-2 (Master Thesis)

This project investigates how occupations are represented in the embedding space of GPT-2 and whether these representations exhibit gender bias. It compares model-derived bias scores with real-world labor statistics from the US Bureau of Labor Statistics (BLS).

## About the Project

The analysis explores occupational gender bias using three primary dimensions:

1. **Local Semantic Proximity**: Evaluating the distance between occupations and gender-specific terms.
2. **Geometric Alignment**: Projecting occupation embeddings onto a defined "gender axis" (man vs. woman).
3. **Contextual Organizational Structure**: Using KMeans clustering and PCA to visualize how the model clusters occupations based on representative sentences.

The study utilizes a dataset of 100 occupations paired with BLS demographic data and contextual sentences from the Bias in Bios dataset.

## Quick Start

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run Analysis**
   Execute the Jupyter Notebooks in `notebooks/` following their numeric order:
   - `01-02`: Data preparation and embedding generation.
   - `03-05`: Analysis via cosine similarity, correlation with BLS data, and gender projection.
   - `06-07`: Semantic clustering using contextualized GPT-2 embeddings.

## Project Structure

- `src/`: Shared utilities, model loading logic, and project configurations.
- `notebooks/`: Sequential pipeline for data processing and analysis.
- `data/`: Raw source files (BLS data) and processed datasets (ignored by Git).
- `results/`: Output directory for generated figures and computed data files (ignored by Git).

## Key Methods

- **Cosine Similarity**: Assessing semantic distance between occupations and 'he', 'she', 'man', 'woman'.
- **Gender Projection**: Measuring alignment along a vector defined by gender anchor terms.
- **Semantic Clustering**: Analyzing the semantic grouping of occupations through average contextual embeddings.
