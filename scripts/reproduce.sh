#!/bin/bash
# Reproduction script for Autonomous AI Researcher experiments

echo "=== REPRODUCTION SEQUENCE START ==="

# 1. Verification of environment imports
echo "Verifying local python environment modules..."
python -c "
import torch
import langgraph
import chromadb
import trafilatura
import pandas
print('All libraries verified and loaded successfully!')
" || { echo "Failed to import core libraries. Run scripts/setup_env.sh first."; exit 1; }

# 2. Setup benchmark directories and load miniature datasets
echo "Initializing HotpotQA evaluation files..."
python -c "
from src.evaluation.benchmarks.loaders import HotpotQALoader
loader = HotpotQALoader()
loader.load(n_samples=5)
print('HotpotQA loader successfully verified!')
"

# 3. Trigger full experiments evaluation sweep
echo "Triggering complete batch experiment benchmark runner sweep..."
python scripts/run_all_experiments.py --benchmark hotpotqa --samples 3 --output_dir logs/experiments

# 4. Generate comparative charts
echo "Building analysis aggregates comparison dataframes..."
python -c "
from src.evaluation.analysis import ResultsAnalyzer
analyzer = ResultsAnalyzer()
df = analyzer.load_all_runs()
if not df.empty:
    comp_df = analyzer.build_comparison_table(df)
    print(comp_df.to_string())
    analyzer.plot_bar_comparison(df, 'mean_f1_score', 'logs/experiments/f1_comparison.png')
    analyzer.plot_bar_comparison(df, 'mean_citation_precision', 'logs/experiments/citation_precision_comparison.png')
    print('Comparative performance plots saved to logs/experiments/')
else:
    print('No previous experiment output files found for plotting.')
"

echo "=== REPRODUCTION SEQUENCE COMPLETED ==="
