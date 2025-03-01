# Master Thesis: LLMs Bias Analysis

## Project Structure

```
master_thesis_llms_bias/
├── data/                       # All data files
│   ├── raw/                    # Raw data
│   │   └── bios_*.pkl          # BIB dataset files
│   └── processed/              # Processed data
├── experiments/                # Experiment code
│   ├── gpt2/                   # GPT-2 related experiments
│   │   ├── exp_1.py            # Experiment 1 code
│   │   └── ...                 # Other experiments
│   └── other_models/           # Other model experiments
├── notebooks/                  # Jupyter notebooks
│   └── exp_1.ipynb             # Experiment 1 notebook
├── results/                    # Experiment results
│   ├── figures/                # Charts and figures
│   └── tables/                 # Table data
├── src/                        # Source code
│   ├── data_processing/        # Data processing code
│   ├── models/                 # Model-related code
│   └── evaluation/             # Evaluation metrics code
├── tests/                      # Test code
├── .gitignore                  # Git ignore file
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

## Project Description

This project investigates bias issues in Large Language Models (LLMs), with a particular focus on gender bias. The project utilizes the BIB dataset for experiments, conducting multiple analyses to quantify and analyze biases within the models.

## Installation

```bash
pip install -r requirements.txt
```

## Dataset

The project uses the BIB (Bias in Bios) dataset, which contains biographical texts of various professions, used for analyzing gender bias.

## Experiments

- exp_1: Analysis of gender bias in occupational word embeddings using the GPT-2 model
