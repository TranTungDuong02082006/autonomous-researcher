import sys
import os
import argparse
import yaml
import json
import logging

# Add src/ to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.benchmarks.loaders import HotpotQALoader, GAIALoader
from src.evaluation.runner import BenchmarkRunner

def main():
    parser = argparse.ArgumentParser(description="Evaluate Agent Performance over benchmarks")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config YAML")
    parser.add_argument("--benchmark", type=str, default="hotpotqa", choices=["hotpotqa", "gaia"], help="Target benchmark to evaluate")
    parser.add_argument("--samples", type=int, default=3, help="Number of benchmark questions to evaluate")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Path to save evaluation output results")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("run_benchmark")

    logger.info(f"Loading config from {args.config}...")
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 1. Load benchmark questions
    logger.info(f"Loading benchmark: {args.benchmark}...")
    if args.benchmark == "hotpotqa":
        loader = HotpotQALoader()
        questions = loader.load(n_samples=args.samples)
    elif args.benchmark == "gaia":
        loader = GAIALoader()
        questions = loader.load(n_samples=args.samples)
    else:
        logger.error(f"Unsupported benchmark: {args.benchmark}")
        sys.exit(1)

    logger.info(f"Successfully loaded {len(questions)} evaluation samples.")

    # 2. Run runner
    runner = BenchmarkRunner(config=config, questions=questions)
    summary_results = runner.run(max_workers=2)

    # 3. Export results to file
    logger.info(f"Saving compiled benchmark stats results to {args.output}...")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, ensure_ascii=False, indent=2)

    print("\n" + "="*50)
    print(f"BENCHMARK COMPLETED: {args.benchmark.upper()}")
    print("="*50)
    for k, v in summary_results.items():
        if isinstance(v, float):
            print(f"{k:35}: {v:.4f}")
        else:
            print(f"{k:35}: {v}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
