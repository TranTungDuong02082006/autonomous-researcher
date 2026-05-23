import os
import json
import logging
import glob
from typing import Dict, Any, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class ResultsAnalyzer:
    def __init__(self, results_dir: str = "logs/experiments"):
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

    def load_all_runs(self) -> pd.DataFrame:
        """Find and parse all individual JSON results files from experiment runs."""
        json_files = glob.glob(os.path.join(self.results_dir, "*_results.json"))
        
        runs = []
        for file in json_files:
            filename = os.path.basename(file)
            if filename == "experiments_comparison.json":
                continue
            exp_name = filename.replace("_results.json", "")
            
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data["experiment"] = exp_name
                runs.append(data)
            except Exception as e:
                logger.error(f"Failed to load result file {file}: {e}")
                
        return pd.DataFrame(runs)

    def build_comparison_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create structured comparison table grouped by experiment name."""
        if df.empty:
            return pd.DataFrame()
            
        metrics = [
            "experiment", "mean_f1_score", "mean_citation_precision", 
            "mean_overall_judge", "mean_step_count", "success_rate"
        ]
        columns_to_keep = [m for m in metrics if m in df.columns]
        return df[columns_to_keep].set_index("experiment")

    def plot_bar_comparison(self, df: pd.DataFrame, metric: str, save_path: str):
        """Plot and save comparative bar charts for different experiment settings."""
        if df.empty or metric not in df.columns:
            return
            
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x="experiment", y=metric, palette="viridis")
        plt.title(f"Comparative Evaluation: {metric.replace('_', ' ').title()}")
        plt.ylabel(metric.replace("_", " ").title())
        plt.xlabel("Experiment Variant")
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Comparison plot saved successfully to {save_path}")

    def error_analysis(self, trace_files: List[str]) -> Dict[str, int]:
        """Categorize failure modes across executed runs by scanning final error logs."""
        categories = {
            "timeout": 0,
            "max_steps_exceeded": 0,
            "loop_detected": 0,
            "citation_failure": 0,
            "llm_error": 0
        }
        
        for file in trace_files:
            if not os.path.exists(file):
                continue
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                # Scan steps for error patterns
                for step in data.get("steps", []):
                    error_log = step.get("output", {}).get("error_log", [])
                    for err in error_log:
                        err_lower = err.lower()
                        if "timeout" in err_lower:
                            categories["timeout"] += 1
                        elif "max steps" in err_lower:
                            categories["max_steps_exceeded"] += 1
                        elif "loop" in err_lower:
                            categories["loop_detected"] += 1
                        elif "citation" in err_lower:
                            categories["citation_failure"] += 1
                        else:
                            categories["llm_error"] += 1
            except Exception as e:
                logger.error(f"Error parsing trace file {file}: {e}")
                
        return categories
