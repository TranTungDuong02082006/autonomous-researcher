import sys
import os
import argparse
import yaml
import logging
from typing import Dict, Any

# Add src/ to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.models.llm_server import LLMClient
from src.tools.web_search import WebSearchTool
from src.tools.scraper import WebScraper
from src.memory.vector_store import EvidenceMemory
from src.tools.summarizer import Summarizer
from src.utils.tracer import TraceLogger

from src.agents.planner import Planner
from src.agents.researcher import Researcher
from src.agents.reviewer import Reviewer
from src.agents.writer import Writer
from src.graph.build_graph import build_research_graph

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    parser = argparse.ArgumentParser(description="Run the Autonomous AI Researcher agentic pipeline")
    parser.add_argument("--query", type=str, required=True, help="Scientific/Academic topic to research")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to YAML config file")
    parser.add_argument("--output", type=str, default="output_report.md", help="Filename to output final Markdown report")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("run_agent")

    logger.info(f"Loading config from {args.config}...")
    if not os.path.exists(args.config):
        logger.error(f"Config file not found at {args.config}")
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 1. Initialize Tracing
    logging_config = config.get("logging", {})
    trace_dir = logging_config.get("trace_dir", "logs/traces")
    tracer = TraceLogger(trace_dir=trace_dir)

    # 2. Initialize LLM Clients
    llm_config = config.get("llm", {})
    logger.info("Initializing LLM client wrapper...")
    llm_client = LLMClient(
        model_name=llm_config.get("model"),
        backend=llm_config.get("backend"),
        quantization=llm_config.get("quantization"),
        api_base=llm_config.get("api_base"),
        api_key=llm_config.get("api_key"),
        temperature=llm_config.get("temperature", 0.7),
        max_tokens=llm_config.get("max_tokens", 4096)
    )

    # 3. Initialize Tools and Memory Vector Store
    logger.info("Initializing tools and local memory systems...")
    tools_config = config.get("tools", {})
    search_tool = WebSearchTool(
        provider=tools_config.get("search_provider", "tavily"),
        max_results=tools_config.get("search_max_results", 5)
    )

    scraper = WebScraper(
        timeout=tools_config.get("scrape_timeout", 15),
        chunk_size=config.get("memory", {}).get("chunk_size", 512),
        chunk_overlap=config.get("memory", {}).get("chunk_overlap", 50)
    )

    memory_config = config.get("memory", {})
    memory = EvidenceMemory(
        embedding_model=memory_config.get("embedding_model", "all-MiniLM-L6-v2")
    )

    summarizer = Summarizer(llm_client=llm_client)

    # 4. Initialize Agent components
    logger.info("Initializing planners, peer reviewers, and writers...")
    planner = Planner(llm_client=llm_client, tracer=tracer)
    researcher = Researcher(
        llm_client=llm_client,
        search_tool=search_tool,
        scraper=scraper,
        memory=memory,
        summarizer=summarizer,
        tracer=tracer
    )
    reviewer = Reviewer(llm_client=llm_client, tracer=tracer)
    writer = Writer(llm_client=llm_client, tracer=tracer)

    # 5. Compile the LangGraph
    logger.info("Assembling LangGraph workflow...")
    agent_config = config.get("agent", {})
    max_reflection_loops = agent_config.get("max_reflection_loops", 3)
    graph = build_research_graph(
        planner=planner,
        researcher=researcher,
        reviewer=reviewer,
        writer=writer,
        max_reflection_loops=max_reflection_loops
    )

    # 6. Execute pipeline
    initial_state = {
        "user_query": args.query,
        "config": config,
        "research_plan": [],
        "current_task_idx": 0,
        "search_history": [],
        "collected_evidence": [],
        "review_feedback": None,
        "reflection_count": 0,
        "draft_sections": {},
        "final_report": None,
        "citations": [],
        "step_count": 0,
        "status": "planning",
        "error_log": []
    }

    logger.info(f"Triggering research workflow on: '{args.query}'...")
    try:
        final_state = graph.invoke(initial_state)
        
        report = final_state.get("final_report", "")
        if report:
            logger.info("Research process completed successfully!")
            print("\n" + "="*50 + "\n")
            print(report)
            print("\n" + "="*50 + "\n")

            # Write final report file
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(report)
            logger.info(f"Saved complete Markdown research report to {args.output}")
        else:
            logger.warning("Pipeline executed but no report was generated.")
            
    except Exception as e:
        logger.exception(f"Fatal crash during workflow execution: {e}")
        
    finally:
        # Export JSON trace
        trace_path = tracer.export(args.query)
        logger.info(f"Trace file compiled at: {trace_path}")

if __name__ == "__main__":
    main()
