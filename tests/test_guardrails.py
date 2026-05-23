import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.graph.guardrails import check_max_steps, check_repeated_queries, check_evidence_sufficiency
from src.graph.state import SearchQuery, EvidenceModel

def test_check_max_steps():
    state_ok = {
        "step_count": 5,
        "config": {"agent": {"max_steps": 10}}
    }
    state_triggered = {
        "step_count": 12,
        "config": {"agent": {"max_steps": 10}}
    }
    
    assert check_max_steps(state_ok) is False
    assert check_max_steps(state_triggered) is True

def test_check_repeated_queries():
    state_no_history = {
        "search_history": []
    }
    state_distinct = {
        "search_history": [
            SearchQuery(query="Transformer architectures paper google", timestamp=""),
            SearchQuery(query="Qwen LLM Alibaba benchmarks", timestamp="")
        ]
    }
    state_similar = {
        "search_history": [
            SearchQuery(query="Transformer architectures paper google", timestamp=""),
            SearchQuery(query="transformer architecture paper google", timestamp="")
        ]
    }

    assert check_repeated_queries(state_no_history) is False
    assert check_repeated_queries(state_distinct) is False
    assert check_repeated_queries(state_similar) is True

def test_check_evidence_sufficiency():
    state_low = {
        "collected_evidence": [EvidenceModel(text="A", source_url="B", title="C", score=1.0)] * 5
    }
    state_sufficient = {
        "collected_evidence": [EvidenceModel(text="A", source_url="B", title="C", score=1.0)] * 20
    }

    assert check_evidence_sufficiency(state_low) is False
    assert check_evidence_sufficiency(state_sufficient) is True
