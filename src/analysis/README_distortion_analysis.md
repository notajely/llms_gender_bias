# Representational Distortion Analysis

This script identifies occupations with significant representational distortion in language models, where the model's gender projection scores poorly correlate with real-world female ratios. It calculates distortion scores and identifies the most distorted occupations.

## Features

- Loads occupation data with BLS gender ratios
- Loads gender projection scores from the model
- Merges these datasets
- Calculates distortion scores using regression residuals
- Identifies occupations with significant distortion
- Visualizes the results
- Saves the results to CSV files

## Usage

Run the script from the project root directory:

```bash
python -m src.analysis.representational_distortion_analysis
```

## Methodology

The script uses the following methodology:

1. **Linear Regression Model**:
   - Fits a linear regression model to predict projection scores from female ratios
   - Calculates the expected projection score for each occupation based on its female ratio

2. **Distortion Score Calculation**:
   - Calculates the residual (actual projection score - expected projection score)
   - Standardizes the residuals to get Z-scores
   - Uses the absolute Z-score as the distortion magnitude
   - Determines the distortion direction (male-biased or female-biased)

3. **Significant Distortion Identification**:
   - Identifies occupations with absolute Z-scores above a threshold (default: 1.5)
   - Sorts occupations by distortion magnitude

## Outputs

The script generates the following outputs in the `results/distortion_analysis/` directory:

1. **CSV Files**:
   - `all_occupation_distortion_scores.csv`: Contains distortion scores for all occupations
   - `significant_distortion_occupations.csv`: Contains only occupations with significant distortion

2. **Visualizations**:
   - `distortion_analysis_scatter.png`: Scatter plot showing all occupations, with significantly distorted occupations highlighted
   - `top_distorted_occupations.png`: Bar plot of the most distorted occupations
   - `distortion_by_category.png`: Box plot showing the distribution of distortion by stereotype category

## Interpretation

- **Male-biased distortion**: The occupation is represented as more masculine in the model than would be expected based on real-world gender ratios
- **Female-biased distortion**: The occupation is represented as more feminine in the model than would be expected based on real-world gender ratios

The magnitude of distortion (Z-score) indicates how extreme the distortion is relative to other occupations.

## Example Results

The script will print a summary of the analysis, including:
- Total number of occupations analyzed
- Number of occupations with significant distortion
- Top 10 most distorted occupations

Example output:
```
Distortion Analysis Summary:
Total occupations analyzed: 250
Occupations with significant distortion: 42

Top 10 most distorted occupations:
  nurse: 3.25 SD more feminine than expected
  engineer: 2.98 SD more masculine than expected
  ...
```
