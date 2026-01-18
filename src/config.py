from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data Directories
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# Results Directories
RESULTS_DIR = PROJECT_ROOT / 'results'
FIGURES_DIR = RESULTS_DIR / 'figures'
RESULTS_DATA_DIR = RESULTS_DIR / 'data'

# Ensure directories exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, RESULTS_DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Model Settings
MODEL_NAME = 'gpt2'
