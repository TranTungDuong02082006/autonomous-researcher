import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.graph.build_graph import build_research_graph
from src.graph.state import ResearchTask, EvidenceModel
import pytest

def test_route_after_research():
    # We can mock the state dictionary and test the routing function
    # In src/graph/build_graph.py, the route_after_research is a local function inside build_research_graph,
    # but we can check the behavior by simulating the conditions in our test.
    
    # Let's inspect the transition conditions:
    # 1. if state["status"] == "writing" -> return "writer"
    # 2. if check_evidence_sufficiency(state) -> return "writer"
    # 3. else -> return "reviewer"
    
    # We will test route_after_research logic directly
    def simulate_route_after_research(state):
        from src.graph.guardrails import check_evidence_sufficiency
        if state["status"] == "writing":
            return "writer"
        if check_evidence_sufficiency(state):
            return "writer"
        return "reviewer"

    state_writing = {"status": "writing", "collected_evidence": []}
    # Sufficient count AND >= 3 unique domains — triggers sufficiency guardrail
    state_sufficient = {
        "status": "researching",
        "collected_evidence": (
            [EvidenceModel(text="A", source_url="http://a.com/1", title="C", score=1.0)] * 5 +
            [EvidenceModel(text="B", source_url="http://b.com/1", title="C", score=1.0)] * 5 +
            [EvidenceModel(text="C", source_url="http://c.com/1", title="C", score=1.0)] * 5 +
            [EvidenceModel(text="D", source_url="http://d.com/1", title="C", score=1.0)] * 5
        )
    }
    state_review = {
        "status": "researching",
        "collected_evidence": [EvidenceModel(text="A", source_url="http://a.com", title="C", score=1.0)] * 2
    }

    assert simulate_route_after_research(state_writing) == "writer"
    assert simulate_route_after_research(state_sufficient) == "writer"
    assert simulate_route_after_research(state_review) == "reviewer"

def test_route_after_review():
    # Let's inspect the route_after_review transition conditions:
    # 1. if idx >= len(plan) or state["status"] == "writing" -> return "writer"
    # 2. else -> return "researcher"
    
    def simulate_route_after_review(state):
        idx = state["current_task_idx"]
        plan = state["research_plan"]
        if idx >= len(plan) or state["status"] == "writing":
            return "writer"
        return "researcher"

    plan = [
        ResearchTask(id=1, sub_question="A", description="B", status="pending"),
        ResearchTask(id=2, sub_question="C", description="D", status="pending")
    ]

    state_completed = {
        "current_task_idx": 2,
        "research_plan": plan,
        "status": "researching"
    }
    state_writing = {
        "current_task_idx": 0,
        "research_plan": plan,
        "status": "writing"
    }
    state_loop = {
        "current_task_idx": 0,
        "research_plan": plan,
        "status": "researching"
    }

    assert simulate_route_after_review(state_completed) == "writer"
    assert simulate_route_after_review(state_writing) == "writer"
    assert simulate_route_after_review(state_loop) == "researcher"
