#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Clustering Pipeline

This script runs the semantic clustering pipeline with two options:

1. Full pipeline (default):
   a. Extract sentences from the BiB dataset
   b. Generate contextualized embeddings
   c. Perform KMeans clustering and PCA
   d. Analyze the clustering results

2. Direct analysis using existing bib_bias_cluster_analysis.csv:
   a. Analyze the clustering results directly from the CSV file
"""

import os
import sys
import time
import argparse

def run_script(script_path, description):
    """
    Run a Python script and print its output.

    Args:
        script_path: Path to the script
        description: Description of the script
    """
    print(f"\n{'=' * 80}")
    print(f"Running {description}...")
    print(f"{'=' * 80}\n")

    start_time = time.time()
    exit_code = os.system(f"python {script_path}")
    end_time = time.time()

    if exit_code != 0:
        print(f"\nError running {script_path}. Exit code: {exit_code}")
        sys.exit(exit_code)

    print(f"\n{'-' * 80}")
    print(f"Completed {description} in {end_time - start_time:.2f} seconds")
    print(f"{'-' * 80}\n")

def main():
    """Main function to run the pipeline."""
    parser = argparse.ArgumentParser(description="Run the semantic clustering pipeline")
    parser.add_argument("--direct-analysis", action="store_true",
                        help="Skip the full pipeline and directly analyze the bib_bias_cluster_analysis.csv file")
    parser.add_argument("--skip-extract", action="store_true", help="Skip sentence extraction")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding generation")
    parser.add_argument("--skip-clustering", action="store_true", help="Skip clustering analysis")
    parser.add_argument("--skip-stereotype", action="store_true", help="Skip stereotype analysis")
    args = parser.parse_args()

    # Create results directory
    os.makedirs("results/semantic_clustering", exist_ok=True)

    if args.direct_analysis:
        print("Running direct analysis using existing bib_bias_cluster_analysis.csv file")
        run_script(
            "src/analysis/cluster_stereotype_analysis.py",
            "Stereotype Analysis from Existing Data"
        )
    else:
        # Step 1: Extract sentences
        if not args.skip_extract:
            run_script(
                "src/data_processing/extract_bib_sentences.py",
                "Sentence Extraction"
            )
        else:
            print("Skipping sentence extraction")

        # Step 2: Generate embeddings
        if not args.skip_embeddings:
            run_script(
                "src/analysis/generate_contextualized_embeddings.py",
                "Embedding Generation"
            )
        else:
            print("Skipping embedding generation")

        # Step 3: Perform clustering
        if not args.skip_clustering:
            run_script(
                "src/analysis/semantic_clustering_analysis.py",
                "Clustering Analysis"
            )
        else:
            print("Skipping clustering analysis")

        # Step 4: Analyze stereotypes
        if not args.skip_stereotype:
            run_script(
                "src/analysis/cluster_stereotype_analysis.py",
                "Stereotype Analysis"
            )
        else:
            print("Skipping stereotype analysis")

    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()
