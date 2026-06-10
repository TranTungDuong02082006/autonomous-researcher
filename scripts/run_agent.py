import sys
import os
import argparse
import yaml
import logging
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# HuggingFace model cache: defaults to models/ inside project root
# Override by setting HF_HOME in your .env file
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_hf_home = os.environ.get("HF_HOME", os.path.join(_project_root, "models"))
os.environ["HF_HOME"] = _hf_home
os.environ["TRANSFORMERS_CACHE"] = _hf_home

# Add src/ to path
sys.path.append(_project_root)

from src.models.llm_server import LLMClient
from src.tools.web_search import WebSearchTool
from src.tools.scraper import WebScraper
from src.memory.vector_store import EvidenceMemory
from src.tools.summarizer import Summarizer
from src.utils.tracer import TraceLogger
from src.models.fine_tuned_models import FineTunedReviewer, FineTunedReranker

from src.agents.planner import Planner
from src.agents.researcher import Researcher
from src.agents.reviewer import Reviewer
from src.agents.writer import Writer
from src.graph.build_graph import build_research_graph

def setup_logging():
    try:
        sys.stdout.reconfigure(errors='replace')
    except AttributeError:
        pass

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
    logger.info("Initializing LLM client wrappers with Model Routing...")
    
    # --- BỘ ĐỊNH TUYẾN MÔ HÌNH CHUẨN REPO: TÔN TRỌNG YAML TUYỆT ĐỐI ---
    planner_model = llm_config.get("model", "Qwen/Qwen2.5-14B-Instruct")
    reviewer_model = planner_model
    writer_model = planner_model
    light_model = llm_config.get("model", "Qwen/Qwen2.5-14B-Instruct")

    # High-quality client for Planner
    planner_llm = LLMClient(
        model_name=planner_model,
        backend=llm_config.get("backend"),
        quantization=llm_config.get("quantization"),
        api_base=llm_config.get("api_base"),
        api_key=llm_config.get("api_key"),
        temperature=llm_config.get("temperature", 0.7),
        max_tokens=llm_config.get("max_tokens", 4096)
    )

    # Light client for Summarizer and Query Expansion (Researcher)
    light_llm = LLMClient(
        model_name=light_model,
        backend=llm_config.get("backend"),
        quantization=llm_config.get("quantization"),
        api_base=llm_config.get("api_base"),
        api_key=llm_config.get("api_key"),
        temperature=0.3,
        max_tokens=1024
    )

    # Reviewer runs on 70B model
    reviewer_llm = LLMClient(
        model_name=reviewer_model,
        backend=llm_config.get("backend"),
        quantization=llm_config.get("quantization"),
        api_base=llm_config.get("api_base"),
        api_key=llm_config.get("api_key"),
        temperature=0.3,
        max_tokens=4096
    )

    # Writer runs on 70B model
    writer_llm = LLMClient(
        model_name=writer_model,
        backend=llm_config.get("backend"),
        quantization=llm_config.get("quantization"),
        api_base=llm_config.get("api_base"),
        api_key=llm_config.get("api_key"),
        temperature=0.5,
        max_tokens=4096
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

    fine_tuned_config = config.get("fine_tuned", {})

    # Initialize Reranker if enabled
    reranker = None
    reranker_config = fine_tuned_config.get("reranker", {})
    if reranker_config.get("enabled", False):
        model_path = reranker_config.get("model_path", "reranker model")
        logger.info(f"Fine-tuned Reranker is enabled. Loading CrossEncoder from: {model_path}")
        try:
            reranker = FineTunedReranker(model_path=model_path)
        except Exception as e:
            logger.warning(f"Failed to load FineTunedReranker: {e}. Running without reranking.")

    # Initialize Reviewer adapter if enabled
    fine_tuned_reviewer = None
    reviewer_config = fine_tuned_config.get("reviewer", {})
    if reviewer_config.get("enabled", False):
        base_model = reviewer_config.get("base_model", "Qwen/Qwen2.5-14B-Instruct")
        adapter_path = reviewer_config.get("adapter_path", "reviewer lora")
        logger.info(f"Fine-tuned Reviewer is enabled. Loading adapter from {adapter_path} (Base: {base_model})")
        try:
            fine_tuned_reviewer = FineTunedReviewer(base_model_name=base_model, adapter_path=adapter_path)
        except Exception as e:
            logger.warning(f"Failed to load FineTunedReviewer adapter: {e}. Peer-reviewer will use default LLM server API.")

    memory_config = config.get("memory", {})
    memory = EvidenceMemory(
        embedding_model=memory_config.get("embedding_model", "BAAI/bge-m3"),
        reranker=reranker
    )

    # Summarizer routed to the light LLM client (Llama 8B)
    summarizer = Summarizer(llm_client=light_llm)

    # 4. Initialize Agent components
    logger.info("Initializing planners, peer reviewers, and writers...")
    planner = Planner(llm_client=planner_llm, tracer=tracer)
    
    # Researcher (Query Expansion) routed to the light LLM client (Llama 8B)
    researcher = Researcher(
        llm_client=light_llm,
        search_tool=search_tool,
        scraper=scraper,
        memory=memory,
        summarizer=summarizer,
        tracer=tracer
    )
    reviewer = Reviewer(llm_client=reviewer_llm, tracer=tracer, fine_tuned_reviewer=fine_tuned_reviewer)
    writer = Writer(llm_client=writer_llm, tracer=tracer)

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
        "error_log": [],
        "reviewer_decisions": []
    }

    logger.info(f"Triggering research workflow on: '{args.query}'...")
    try:
        final_state = graph.invoke(initial_state)
        
        report = final_state.get("final_report", "")
        if report:
            logger.info("Research process completed successfully!")
            print("\n" + "="*50 + "\n")
            try:
                print(report)
            except UnicodeEncodeError:
                # Safe print fallback for consoles that do not support unicode characters (e.g. Windows cp1252)
                encoding = sys.stdout.encoding or 'utf-8'
                print(report.encode(encoding, errors='replace').decode(encoding))
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
