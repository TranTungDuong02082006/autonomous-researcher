import re
import logging
from collections import Counter
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def normalize_text(text: str) -> str:
    """Lower text, strip punctuation, articles, and extra whitespace."""
    text = text.lower().strip()
    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove extra spaces
    text = " ".join(text.split())
    return text

def answer_f1(predicted: str, ground_truth: str) -> float:
    """Compute token-level F1 score for question-answering evaluation."""
    pred_tokens = normalize_text(predicted).split()
    gt_tokens = normalize_text(ground_truth).split()
    
    if not pred_tokens or not gt_tokens:
        return 1.0 if pred_tokens == gt_tokens else 0.0
        
    common_tokens = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common_tokens.values())
    
    if num_same == 0:
        return 0.0
        
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return float(f1)

def task_success_rate(traces: List[Dict[str, Any]]) -> float:
    """Percentage of runs completed successfully without final state errors."""
    if not traces:
        return 0.0
    success = 0
    for trace in traces:
        steps = trace.get("steps", [])
        if not steps:
            continue
        final_step = steps[-1]
        # Check if the final step finished without erroring out
        if final_step.get("output", {}).get("status") != "error":
            success += 1
    return float(success / len(traces))

def avg_steps(traces: List[Dict[str, Any]]) -> float:
    """Calculate the average number of steps taken across runs."""
    if not traces:
        return 0.0
    total_steps = sum(len(trace.get("steps", [])) for trace in traces)
    return float(total_steps / len(traces))

def citation_precision(report_body: str, citations: List[Dict[str, Any]]) -> float:
    """
    Evaluates whether listed citations are actually referenced within the report text body.
    Returns the percentage of cited URLs that are actively mapped to a '[i]' bracket in the text.
    """
    if not citations:
        return 1.0
        
    valid_citations = 0
    for cit in citations:
        idx = cit.get("index")
        bracket_pattern = f"\\[{idx}\\]"
        if re.search(bracket_pattern, report_body):
            valid_citations += 1
            
    return float(valid_citations / len(citations))
