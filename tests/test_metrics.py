import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.metrics import answer_f1, citation_precision

def test_answer_f1():
    # Exact match
    assert answer_f1("The model size is 175B parameters.", "175B parameters") > 0.6
    # Zero match
    assert answer_f1("Apple juice is sweet", "Transformers introduced self-attention") == 0.0
    # Partial match
    assert answer_f1("Transformer models perform well", "Transformer model architecture") > 0.2

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
