import json
import logging
import os
import random
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

class ReviewerDatasetBuilder:
    def __init__(self, output_dir: str = "data/training/reviewer"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_synthetic_from_traces(self, trace_files: List[str], n_samples: int = 500):
        """Parse TraceLogger execution logs to generate positive/negative evidence reviews."""
        samples = []
        for file in trace_files:
            if not os.path.exists(file):
                continue
            try:
                with open(file, "r", encoding="utf-8") as f:
                    trace_data = json.load(f)
                
                # Extract reviewer steps
                for step in trace_data.get("steps", []):
                    if step.get("agent_name") == "Reviewer" and " Peer review" in step.get("action", ""):
                        # Extract claim context and evaluation
                        inp = step.get("input", "")
                        out = step.get("output", {})
                        
                        # Re-format as standard conversational message pair for SFT SFTTrainer
                        chat_format = {
                            "messages": [
                                {"role": "system", "content": "You are a scientific peer reviewer checking evidence sufficiency."},
                                {"role": "user", "content": f"Assess this evidence: {inp}"},
                                {"role": "assistant", "content": json.dumps(out)}
                            ]
                        }
                        samples.append(chat_format)
            except Exception as e:
                logger.error(f"Failed to process trace file {file}: {e}")

        # Add generic synthetic QA samples to ensure dataset is populated
        if len(samples) < 5:
            logger.warning("No sufficient trace files found. Generating dummy reviewer SFT samples.")
            dummy_samples = [
                {
                    "messages": [
                        {"role": "system", "content": "You are a scientific peer reviewer checking evidence sufficiency."},
                        {"role": "user", "content": "Assess this evidence: Sub-question: 'Context length of Llama 3 8B'. Evidence: [1] Llama 3 has a base context window of 8k tokens."},
                        {"role": "assistant", "content": json.dumps({"sufficient": True, "missing_info": None, "confidence": 0.95, "findings": "Llama 3 8B has an 8k token context window."})}
                    ]
                },
                {
                    "messages": [
                        {"role": "system", "content": "You are a scientific peer reviewer checking evidence sufficiency."},
                        {"role": "user", "content": "Assess this evidence: Sub-question: 'Context length of Claude 3 Opus'. Evidence: [1] Claude 3 family was released in 2024 by Anthropic."},
                        {"role": "assistant", "content": json.dumps({"sufficient": False, "missing_info": "The exact token capacity or context window length of Claude 3 Opus is not mentioned in the source.", "confidence": 0.9, "findings": "No window capacity found."})}
                    ]
                }
            ]
            samples.extend(dummy_samples)

        # Train/Val/Test Split
        random.shuffle(samples)
        split_idx = int(0.8 * len(samples))
        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]

        self._save_jsonl(train_samples, os.path.join(self.output_dir, "train.jsonl"))
        self._save_jsonl(val_samples, os.path.join(self.output_dir, "val.jsonl"))
        logger.info(f"Saved {len(train_samples)} training samples and {len(val_samples)} validation samples to {self.output_dir}")

    def _save_jsonl(self, data: List[Dict[str, Any]], filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


class RerankerDatasetBuilder:
    def __init__(self, output_dir: str = "data/training/reranker"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_triplets(self):
        """Generate positive and hard negative triplets (query, positive, negative) for contrastive fine-tuning."""
        # Simple sample dataset for contrastive ranking fine-tuning
        triplets = [
            {
                "query": "How large is the context window of Qwen 2.5 14B?",
                "positive": "Qwen 2.5 14B has a context window length of 128k tokens, allowing it to process very long source documents.",
                "negative": "Qwen 2.5 14B is a dense language model trained by Alibaba Group on a rich multilingual pre-training dataset."
            },
            {
                "query": "Who introduced the Transformer architecture?",
                "positive": "The Transformer architecture was introduced by Vaswani et al. from Google in the 2017 paper 'Attention is All You Need'.",
                "negative": "Transformers are widely used in natural language processing and computer vision applications worldwide."
            }
        ]

        filepath = os.path.join(self.output_dir, "triplets.jsonl")
        with open(filepath, "w", encoding="utf-8") as f:
            for item in triplets:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                
        logger.info(f"Generated ranking triplets dataset at {filepath}")
