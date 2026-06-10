import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock
from src.agents.planner import Planner
from src.agents.researcher import Researcher
from src.agents.reviewer import Reviewer
from src.agents.writer import Writer
from src.graph.build_graph import build_research_graph
from src.graph.state import ResearchContext, TemporalScope


def test_planner_guess_language():
    # Test English
    assert Planner._guess_language("What is CRISPR?") == "en"
    # Test Vietnamese
    assert Planner._guess_language("Giá xăng hôm nay ở Việt Nam thế nào?") == "vi"
    # Test Japanese
    assert Planner._guess_language("今日の天気はどうですか") == "ja"
    # Test Korean
    assert Planner._guess_language("오늘 날씨가 어때요?") == "ko"
    # Test Chinese
    assert Planner._guess_language("今天天气怎么样？") == "zh"


def test_graph_compilation_and_interfaces():
    # Mock clients and dependencies
    mock_llm = MagicMock()
    mock_tracer = MagicMock()
    mock_search = MagicMock()
    mock_scraper = MagicMock()
    mock_memory = MagicMock()
    mock_summarizer = MagicMock()

    planner = Planner(llm_client=mock_llm, tracer=mock_tracer)
    researcher = Researcher(
        llm_client=mock_llm,
        search_tool=mock_search,
        scraper=mock_scraper,
        memory=mock_memory,
        summarizer=mock_summarizer,
        tracer=mock_tracer,
        locale_hints_path="configs/locale_hints.yaml"
    )
    reviewer = Reviewer(llm_client=mock_llm, tracer=mock_tracer)
    writer = Writer(llm_client=mock_llm, tracer=mock_tracer)

    # Verify build_research_graph compiles without errors
    compiled_graph = build_research_graph(
        planner=planner,
        researcher=researcher,
        reviewer=reviewer,
        writer=writer
    )
    assert compiled_graph is not None


def test_writer_citation_validation():
    mock_llm = MagicMock()
    mock_tracer = MagicMock()
    writer = Writer(llm_client=mock_llm, tracer=mock_tracer)

    from src.graph.state import Citation, EvidenceModel

    # Mock parameters
    body_with_year = "In April [2026], oil prices dropped [1] but recovered slightly [2]."
    citations = [
        Citation(index=1, url="http://domain1.com", title="First Source"),
        Citation(index=2, url="http://domain2.com", title="Second Source")
    ]
    evidence = [
        EvidenceModel(text="A", source_url="http://domain1.com", title="First Source", score=1.0),
        EvidenceModel(text="B", source_url="http://domain2.com", title="Second Source", score=1.0)
    ]

    # Validate that [2026] is not detected as an invalid citation
    final_body = writer._validate_and_finalize_body(body_with_year, citations, evidence)
    assert "[2026]" in final_body
    assert "[2026⚠]" not in final_body
    assert "[1]" in final_body
    assert "[2]" in final_body


def test_reviewer_node_bounds_guard():
    # Construct a minimal StateGraph or call the reviewer_node directly
    from src.graph.build_graph import build_research_graph
    from src.graph.state import ResearchTask

    mock_planner = MagicMock()
    mock_researcher = MagicMock()
    mock_reviewer = MagicMock()
    mock_writer = MagicMock()

    compiled_graph = build_research_graph(
        planner=mock_planner,
        researcher=mock_researcher,
        reviewer=mock_reviewer,
        writer=mock_writer
    )

    # Reconstruct reviewer_node logic under compilation
    # Let's inspect the behavior of reviewer_node with state current_task_idx >= len(research_plan)
    state = {
        "current_task_idx": 2,
        "research_plan": [
            ResearchTask(id=1, sub_question="Q1", description="D1", status="completed")
        ]
    }

    # Retrieve the node by name from the compiled graph
    reviewer_node_func = compiled_graph.nodes["reviewer"].bound.func
    
    # Run the reviewer node directly and assert transition to writing
    result = reviewer_node_func(state)
    assert result == {"status": "writing"}

