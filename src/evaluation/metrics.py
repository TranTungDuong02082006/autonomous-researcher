import re
import logging
from collections import Counter
from typing import List, Dict, Any, Optional

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
    """Compute token-level F1 score for question-answering evaluation.
    
    For short ground truth answers (< 10 tokens, e.g. 'Sweden'), uses exact
    substring match instead of token-level F1 to avoid meaningless scores
    when comparing a 1-token answer against a 1000+ token report.
    """
    norm_pred = normalize_text(predicted)
    norm_gt = normalize_text(ground_truth)
    
    pred_tokens = norm_pred.split()
    gt_tokens = norm_gt.split()
    
    if not pred_tokens or not gt_tokens:
        return 1.0 if pred_tokens == gt_tokens else 0.0
    
    # For short ground truth answers, use exact substring match
    # to avoid meaningless F1 when comparing "Sweden" vs a full report
    if len(gt_tokens) < 10:
        return 1.0 if norm_gt in norm_pred else 0.0
        
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

def normalize_url(url: str) -> str:
    """Normalize URLs to compare them fairly (strip protocol, www, and trailing slashes)."""
    if not url:
        return ""
    url = url.lower().strip()
    url = re.sub(r'^https?://(www\.)?', '', url)
    url = url.rstrip('/')
    return url

def extract_claims(report_body: str, index: int) -> str:
    """Extract sentences or paragraphs containing the given citation index [index]."""
    # Split by common sentence endings followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', report_body)
    bracket = f"[{index}]"
    matching_sentences = [s.strip() for s in sentences if bracket in s]
    if matching_sentences:
        return " ".join(matching_sentences)
    
    # Fallback to paragraph search if sentence splitting didn't catch it
    paragraphs = report_body.split('\n')
    matching_paragraphs = [p.strip() for p in paragraphs if bracket in p]
    if matching_paragraphs:
        return " ".join(matching_paragraphs)
        
    return ""

def citation_precision(
    report_body: str, 
    citations: List[Dict[str, Any]], 
    collected_evidence: List[Dict[str, Any]] = None,
    judge: Any = None
) -> Optional[float]:
    """
    Evaluates whether listed citations are actually referenced within the report text body
    and optionally verifies using an LLM Judge if the citation claims are fully supported by the evidence.
    Returns the percentage of cited URLs that are actively mapped, and if judge is provided, verified.
    Returns None if both citations and evidence are empty (not applicable).
    """
    # If no citations exist, distinguish between N/A and truly zero
    if not citations:
        if collected_evidence is not None and not collected_evidence:
            return None  # Both empty = not applicable, not a failure
        return 0.0
        
    valid_citations = 0
    for cit in citations:
        idx = cit.get("index")
        url = cit.get("url", "")
        bracket_pattern = f"\\[{idx}\\]"
        
        # 1. Format check: Is it cited in the body?
        if not re.search(bracket_pattern, report_body):
            logger.warning(f"Citation [{idx}] ({url}) is listed but never referenced in report body.")
            continue
            
        # 2. Truth check: If judge and evidence are available, verify the claim
        if judge and collected_evidence:
            # Extract claim
            claim = extract_claims(report_body, idx)
            if not claim:
                claim = report_body
            
            # Find matching evidence text
            evidence_text = ""
            norm_target_url = normalize_url(url)
            for ev in collected_evidence:
                ev_url = ev.get("source_url", "")
                if normalize_url(ev_url) == norm_target_url:
                    evidence_text = ev.get("text", "")
                    break
            
            # If no exact URL match, do a fuzzy/substring match
            if not evidence_text:
                for ev in collected_evidence:
                    ev_url = ev.get("source_url", "")
                    if norm_target_url and normalize_url(ev_url) in norm_target_url:
                        evidence_text = ev.get("text", "")
                        break
                    if ev_url and normalize_url(ev_url) in norm_target_url:
                        evidence_text = ev.get("text", "")
                        break

            if evidence_text:
                try:
                    supported = judge.judge_citation(claim, evidence_text)
                    if supported:
                        valid_citations += 1
                    else:
                        logger.warning(f"Citation [{idx}] claim is NOT supported by the source. Claim: '{claim[:100]}...'")
                except Exception as e:
                    logger.error(f"Failed to verify citation [{idx}] due to judge error: {e}. Falling back to invalid claim to remain strict.")
            else:
                logger.warning(f"Could not find matching evidence in collected_evidence for citation URL: {url}. Rejecting citation as invalid.")
        else:
            # 🩹 BỘ VÁ SIÊU NGHIÊM NGẶT: Nếu có ghi trích dẫn [i] nhưng bộ nhớ rỗng (chưa hề cào web) 
            # thì đây 100% là hành vi bịa đặt tri thức (Hallucination/Fabrication) -> Phạt thẳng về 0!
            if citations and collected_evidence is not None and not collected_evidence:
                logger.warning(f"🚨 Phát hiện Fabrication: Báo cáo chứa trích dẫn [{idx}] nhưng không hề có bằng chứng thu thập!")
                # Không tăng valid_citations, để mặc định không được cộng điểm
            else:
                # Chỉ cho qua nếu cả 2 cùng trống hoặc không kích hoạt chế độ verify
                valid_citations += 1
            
    return float(valid_citations / len(citations)) if citations else 0.0


def ndcg_at_k(relevance_scores: List[int], k: int = 5) -> float:
    """Compute Normalized Discounted Cumulative Gain at k."""
    relevance_scores = relevance_scores[:k]
    if not relevance_scores or sum(relevance_scores) == 0:
        return 0.0
    
    import math
    dcg = 0.0
    for i, rel in enumerate(relevance_scores):
        dcg += (2**rel - 1) / math.log2(i + 2)
        
    idcg = 0.0
    for i, rel in enumerate(sorted(relevance_scores, reverse=True)):
        idcg += (2**rel - 1) / math.log2(i + 2)
        
    if idcg == 0.0:
        return 0.0
    return float(dcg / idcg)


def mrr(relevance_scores: List[int]) -> float:
    """Compute Mean Reciprocal Rank."""
    for i, rel in enumerate(relevance_scores):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0


def is_chunk_relevant(chunk_text: str, supporting_facts: List[str]) -> int:
    """Determine binary relevance of a retrieved chunk compared to ground-truth supporting facts."""
    if not supporting_facts:
        return 0
    norm_chunk = normalize_text(chunk_text)
    for fact in supporting_facts:
        norm_fact = normalize_text(fact)
        if not norm_fact:
            continue
        if norm_fact in norm_chunk or norm_chunk in norm_fact:
            return 1
        fact_words = set(norm_fact.split())
        chunk_words = set(norm_chunk.split())
        fact_words = {w for w in fact_words if len(w) > 2}
        chunk_words = {w for w in chunk_words if len(w) > 2}
        if not fact_words:
            continue
        overlap = fact_words.intersection(chunk_words)
        if len(overlap) / len(fact_words) >= 0.4:
            return 1
    return 0


def is_evidence_sufficient(evidence_chunks: List[Dict[str, Any]], supporting_facts: List[str]) -> bool:
    """Determine ground-truth sufficiency of evidence chunks against supporting facts."""
    if not supporting_facts:
        return True
    matched_facts = 0
    for fact in supporting_facts:
        fact_matched = False
        norm_fact = normalize_text(fact)
        for chunk in evidence_chunks:
            chunk_text = chunk.get("text", "")
            norm_chunk = normalize_text(chunk_text)
            if norm_fact in norm_chunk or norm_chunk in norm_fact:
                fact_matched = True
                break
            fact_words = set(norm_fact.split())
            chunk_words = set(norm_chunk.split())
            fact_words = {w for w in fact_words if len(w) > 2}
            chunk_words = {w for w in chunk_words if len(w) > 2}
            if fact_words:
                overlap = fact_words.intersection(chunk_words)
                if len(overlap) / len(fact_words) >= 0.4:
                    fact_matched = True
                    break
        if fact_matched:
            matched_facts += 1
    return matched_facts == len(supporting_facts)


def reviewer_accuracy(predictions: List[bool], ground_truths: List[bool]) -> float:
    """Compute accuracy of reviewer predictions compared to ground-truths."""
    if not predictions:
        return 1.0 # default to 100% if no predictions made
    correct = sum(1 for p, gt in zip(predictions, ground_truths) if p == gt)
    return float(correct / len(predictions))


def reviewer_f1(predictions: List[bool], ground_truths: List[bool]) -> float:
    """Compute F1 score of reviewer predictions compared to ground-truths."""
    if not predictions:
        return 1.0
    tp = sum(1 for p, gt in zip(predictions, ground_truths) if p and gt)
    fp = sum(1 for p, gt in zip(predictions, ground_truths) if p and not gt)
    fn = sum(1 for p, gt in zip(predictions, ground_truths) if not p and gt)
    
    if tp == 0:
        if fp == 0 and fn == 0:
            return 1.0
        return 0.0
        
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return float(2 * precision * recall / (precision + recall))
