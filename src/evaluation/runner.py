import logging
import time
import concurrent.futures
from typing import Any, Dict, List
import pandas as pd

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
from src.evaluation.benchmarks.loaders import BenchmarkQuestion
from src.evaluation.metrics import answer_f1, citation_precision
from src.evaluation.judge import LLMJudge

logger = logging.getLogger(__name__)

class BenchmarkRunner:
    def __init__(self, config: Dict[str, Any], questions: List[BenchmarkQuestion]):
        self.config = config
        self.questions = questions
        self.results = []

    def run(self, max_workers: int = 2) -> Dict[str, Any]:
        """Run benchmark evaluation over all questions in parallel."""
        logger.info(f"Starting benchmark runner for {len(self.questions)} questions with {max_workers} workers...")
        
        # Instantiate LLMJudge for rating
        llm_config = self.config.get("llm", {})
        judge_client = LLMClient(
            model_name=llm_config.get("model"),
            backend=llm_config.get("backend"),
            quantization=llm_config.get("quantization"),
            api_base=llm_config.get("api_base"),
            api_key=llm_config.get("api_key")
        )
        self.judge = LLMJudge(llm_client=judge_client)

        results_list = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_question = {
                executor.submit(self._run_single, q): q for q in self.questions
            }
            
            for future in concurrent.futures.as_completed(future_to_question):
                q = future_to_question[future]
                try:
                    res = future.result()
                    results_list.append(res)
                    logger.info(f"Completed evaluation for question: '{q.question[:40]}...'")
                except Exception as exc:
                    logger.error(f"Question '{q.question}' generated an exception: {exc}")

        self.results = results_list
        return self._aggregate()

    def _run_single(self, q: BenchmarkQuestion) -> Dict[str, Any]:
        """Run a single benchmark query end-to-end and compute performance metrics."""
        logger.info(f"Evaluating Question: '{q.question}'")
        
        # Create standard local workspace for each thread
        tracer = TraceLogger(trace_dir="logs/traces/eval")
        
        llm_config = self.config.get("llm", {})
        llm_client = LLMClient(
            model_name=llm_config.get("model"),
            backend=llm_config.get("backend"),
            quantization=llm_config.get("quantization"),
            api_base=llm_config.get("api_base"),
            api_key=llm_config.get("api_key"),
            max_tokens=4096
        )

        tools_config = self.config.get("tools", {})
        search_tool = WebSearchTool(
            provider=tools_config.get("search_provider", "tavily"),
            max_results=tools_config.get("search_max_results", 5)
        )

        scraper = WebScraper(
            timeout=tools_config.get("scrape_timeout", 15)
        )

        memory = EvidenceMemory(
            embedding_model=self.config.get("memory", {}).get("embedding_model", "all-MiniLM-L6-v2")
        )

        summarizer = Summarizer(llm_client=llm_client)

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

        graph = build_research_graph(
            planner=planner,
            researcher=researcher,
            reviewer=reviewer,
            writer=writer,
            max_reflection_loops=self.config.get("agent", {}).get("max_reflection_loops", 3)
        )

        initial_state = {
            "user_query": q.question,
            "config": self.config,
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

        # Handle agent run with auto-retry
        final_state = None
        max_retries = 2
        for attempt in range(max_retries):
            try:
                final_state = graph.invoke(initial_state)
                break
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for question: '{q.question}'. Retrying... Error: {e}")
                time.sleep(2)

        if not final_state or not final_state.get("final_report"):
            return {
                "question_id": q.id,
                "f1_score": 0.0,
                "citation_precision": 0.0,
                "judge_comprehensiveness": 1.0,
                "judge_depth": 1.0,
                "overall_judge": 1.0,
                "step_count": initial_state["step_count"],
                "success": False
            }

        report_body = final_state.get("final_report", "")
        citations = [cit.model_dump() for cit in final_state.get("citations", [])]

        # Calculate metrics
        f1 = answer_f1(report_body, q.ground_truth_answer)
        cit_prec = citation_precision(report_body, citations)

        # Rubric scoring
        rubric = (
            "Evaluate the report based on scientific rigour:\n"
            "1. Comprehensiveness: does it answer the question fully?\n"
            "2. Depth: does it provide deep evidence instead of high-level definitions?"
        )
        judge_res = self.judge.judge_report(report_body, q.question, rubric)

        return {
            "question_id": q.id,
            "f1_score": float(f1),
            "citation_precision": float(cit_prec),
            "judge_comprehensiveness": float(judge_res.comprehensiveness),
            "judge_depth": float(judge_res.depth_of_research),
            "overall_judge": float(judge_res.overall_score),
            "step_count": final_state.get("step_count", 0),
            "success": True
        }

    def _aggregate(self) -> Dict[str, Any]:
        """Aggregate per-question scores into a summary result dict."""
        if not self.results:
            return {}

        df = pd.DataFrame(self.results)
        
        summary = {
            "total_questions": len(df),
            "mean_f1_score": float(df["f1_score"].mean()),
            "mean_citation_precision": float(df["citation_precision"].mean()),
            "mean_judge_comprehensiveness": float(df["judge_comprehensiveness"].mean()),
            "mean_judge_depth": float(df["judge_depth"].mean()),
            "mean_overall_judge": float(df["overall_judge"].mean()),
            "mean_step_count": float(df["step_count"].mean()),
            "success_rate": float(df["success"].mean())
        }
        
        logger.info(f"Benchmark Aggregation Completed: {summary}")
        return summary
