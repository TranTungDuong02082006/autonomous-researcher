import os
import glob
import sys
import argparse
import yaml
import json
import logging
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src/ to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.benchmarks.loaders import HotpotQALoader
from src.evaluation.runner import BenchmarkRunner

def main():
    parser = argparse.ArgumentParser(description="Run all experiment configuration benchmarks in batch")
    parser.add_argument("--benchmark", type=str, default="hotpotqa", help="Target benchmark to evaluate")
    parser.add_argument("--samples", type=int, default=2, help="Number of questions to test per experiment")
    parser.add_argument("--output_dir", type=str, default="logs/experiments", help="Folder to save execution outcomes")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("run_all_experiments")

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Discover config files
    config_files = glob.glob("configs/experiments/*.yaml")
    if not config_files:
        logger.error("No experiment configuration files found in configs/experiments/")
        sys.exit(1)

    logger.info(f"Discovered {len(config_files)} experiment YAML files to run: {config_files}")

    # 2. Load questions
    loader = HotpotQALoader()
    questions = loader.load(n_samples=args.samples)

    all_summaries = {}

    # 3. Execution loop
    for config_path in config_files:
        exp_name = os.path.basename(config_path).replace(".yaml", "")
        logger.info(f"\n>>> Running Experiment: {exp_name.upper()} <<<")
        
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        try:
            runner = BenchmarkRunner(config=config, questions=questions)
            summary = runner.run(max_workers=2)
            all_summaries[exp_name] = summary
            
            # Export individual result
            ind_path = os.path.join(args.output_dir, f"{exp_name}_results.json")
            with open(ind_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.exception(f"Experiment {exp_name} failed: {e}")

    # 4. Generate comparison table
    if all_summaries:
        df = pd.DataFrame(all_summaries).T
        comparison_path = os.path.join(args.output_dir, "experiments_comparison.csv")
        df.to_csv(comparison_path)
        logger.info(f"Saved global experiments comparison matrix CSV to {comparison_path}")

        print("\n" + "="*50)
        print("EXPERIMENTS RUN COMPLETE: AGGREGATE SCORES MATRIX")
        print("="*50)
        print(df.to_string())
        print("="*50 + "\n")

if __name__ == "__main__":
    main()
