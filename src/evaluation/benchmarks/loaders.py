import logging
import os
import json
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkQuestion:
    id: str
    question: str
    ground_truth_answer: str
    supporting_facts: List[str]

class HotpotQALoader:
    def __init__(self, data_dir: str = "data/benchmarks"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.filepath = os.path.join(self.data_dir, "hotpotqa_mini.json")

    def load(self, split: str = "dev", n_samples: int = 10) -> List[BenchmarkQuestion]:
        """Load a lightweight subset of HotpotQA questions. If not found locally, create a default set."""
        if os.path.exists(self.filepath):
            logger.info(f"Loading HotpotQA benchmark from local file: {self.filepath}")
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                return [BenchmarkQuestion(**item) for item in data[:n_samples]]

        # Fallback default benchmark subset to prevent system downtime
        logger.warning(f"Benchmark file not found at {self.filepath}. Creating default sample set.")
        default_samples = [
            BenchmarkQuestion(
                id="hp-1",
                question="Which architecture is older: Transformer (attention-based) or LSTM?",
                ground_truth_answer="LSTM (Long Short-Term Memory) was introduced in 1997, while the Transformer was introduced in 2017.",
                supporting_facts=["LSTM was introduced in 1997 by Hochreiter & Schmidhuber.", "Transformer was introduced in 2017 in 'Attention is All You Need'."]
            ),
            BenchmarkQuestion(
                id="hp-2",
                question="What was the primary model size in parameters of the original GPT-3 model described by OpenAI in 2020?",
                ground_truth_answer="175 billion parameters.",
                supporting_facts=["GPT-3 model sizes range from 125M to 175B parameters."]
            ),
            BenchmarkQuestion(
                id="hp-3",
                question="Which optimization trick was designed specifically to accelerate low-rank fine-tuning of LLMs?",
                ground_truth_answer="LoRA (Low-Rank Adaptation).",
                supporting_facts=["LoRA freezes pre-trained weights and adds trainable rank decomposition matrices."]
            )
        ]
        
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump([item.__dict__ for item in default_samples], f, ensure_ascii=False, indent=2)

        return default_samples[:n_samples]

class GAIALoader:
    def __init__(self, data_dir: str = "data/benchmarks"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.filepath = os.path.join(self.data_dir, "gaia_mini.json")

    def load(self, split: str = "validation", n_samples: int = 5) -> List[BenchmarkQuestion]:
        """Load a lightweight subset of GAIA questions. If not found locally, create a default set."""
        if os.path.exists(self.filepath):
            logger.info(f"Loading GAIA benchmark from local file: {self.filepath}")
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                return [BenchmarkQuestion(**item) for item in data[:n_samples]]

        logger.warning(f"Benchmark file not found at {self.filepath}. Creating default sample set.")
        default_samples = [
            BenchmarkQuestion(
                id="gaia-1",
                question="Find the release year of the paper that introduced the 'Qwen' LLM family and identify the lead authors' affiliation.",
                ground_truth_answer="2023, Alibaba Group.",
                supporting_facts=[]
            ),
            BenchmarkQuestion(
                id="gaia-2",
                question="What is the context window length in tokens of the Qwen2.5-72B-Instruct model?",
                ground_truth_answer="128k (128,000) tokens.",
                supporting_facts=[]
            )
        ]
        
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump([item.__dict__ for item in default_samples], f, ensure_ascii=False, indent=2)

        return default_samples[:n_samples]
