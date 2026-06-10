import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.metrics import (
    answer_f1, citation_precision, ndcg_at_k, mrr,
    is_chunk_relevant, is_evidence_sufficient, reviewer_accuracy, reviewer_f1
)

def test_answer_f1():
    # Short ground truth: uses exact substring match (Bug 25 fix)
    assert answer_f1("The model size is 175B parameters.", "175B parameters") == 1.0
    assert answer_f1("Apple juice is sweet", "Transformers introduced self-attention") == 0.0
    assert answer_f1("Transformer models perform well", "Transformer model architecture") == 0.0
    assert answer_f1("The Transformer architecture is great for NLP", "transformer architecture") == 1.0
    long_gt = "The Transformer model architecture introduced self attention mechanism for sequence to sequence tasks"
    assert answer_f1("Transformer architecture self attention mechanism", long_gt) > 0.2

def test_citation_precision():
    citations = [
        {"index": 1, "url": "http://a.com", "title": "A"},
        {"index": 2, "url": "http://b.com", "title": "B"}
    ]
    report_body_both = "We cite [1] and [2] for verification."
    report_body_one = "We cite only [1]."
    report_body_none = "No citations are mentioned."

    assert citation_precision(report_body_both, citations) == 1.0
    assert citation_precision(report_body_one, citations) == 0.5
    assert citation_precision(report_body_none, citations) == 0.0

def test_ranking_metrics():
    # perfect rank
    assert ndcg_at_k([1, 1, 0, 0], k=5) == 1.0
    # worse rank
    assert ndcg_at_k([0, 1, 0, 1], k=5) < 1.0
    # empty
    assert ndcg_at_k([], k=5) == 0.0
    
    # MRR perfect
    assert mrr([1, 0, 0]) == 1.0
    # MRR second
    assert mrr([0, 1, 0]) == 0.5
    # MRR zero
    assert mrr([0, 0, 0]) == 0.0

def test_relevance_helpers():
    facts = ["George R.R. Martin was born on September 20, 1948."]
    chunk_rel = "This book was written by George R.R. Martin, who was born on September 20, 1948."
    chunk_irrel = "Llama 3 is a large language model released by Meta."
    
    assert is_chunk_relevant(chunk_rel, facts) == 1
    assert is_chunk_relevant(chunk_irrel, facts) == 0
    
    evidence_suff = [{"text": chunk_rel}]
    evidence_insuff = [{"text": chunk_irrel}]
    
    assert is_evidence_sufficient(evidence_suff, facts) is True
    assert is_evidence_sufficient(evidence_insuff, facts) is False

def test_reviewer_metrics():
    preds = [True, False, True]
    gts = [True, True, False]
    
    # correct matches: predictions[0] == gts[0] (True == True)
    # total predictions: 3, total correct: 1
    # accuracy = 1 / 3
    assert abs(reviewer_accuracy(preds, gts) - 0.3333333333333333) < 1e-6
    assert reviewer_f1(preds, gts) > 0.0

