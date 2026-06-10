"""
generate_raw_data.py — Chạy 6 thí nghiệm ablation và ghi raw output ra JSONL.

Ghi mỗi câu hỏi ra file JSONL ngay lập tức (append mode) để tránh mất data khi timeout/crash.
Metrics tính ở bước sau (compute_metrics.py hoặc notebook).

Usage:
    PYTHONPATH=. python scripts/generate_raw_data.py [--samples N] [--output_dir DIR] [--exp EXP_NAME]
"""
import os
import sys
import json
import yaml
import logging
import time
import argparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("generate_raw_data")

# ────────────────────────────────────────────────
# Default experiment configs (theo thứ tự ablation)
# ────────────────────────────────────────────────
DEFAULT_CONFIGS = [
    "configs/experiments/exp1_vanilla_llm.yaml",
    "configs/experiments/exp2_rag_baseline.yaml",
    "configs/experiments/exp3_agent_base.yaml",
    "configs/experiments/exp4_agent_ft_reviewer.yaml",
    "configs/experiments/exp5_agent_ft_reranker.yaml",
    "configs/experiments/exp6_agent_full.yaml",
]


def build_graph_from_config(config: dict, shared_llm_client=None):
    """
    Khởi tạo toàn bộ agent stack + LangGraph từ config dict.

    Args:
        config: dict đọc từ YAML experiment config
        shared_llm_client: Nếu cung cấp, TẤT CẢ agents dùng chung client này
                           (tránh load model nhiều lần khi chạy local Kaggle).
                           Nếu None, tạo clients riêng theo config (dùng cho API backends).
    """
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

    llm_cfg   = config.get("llm", {})
    tools_cfg = config.get("tools", {})
    mem_cfg   = config.get("memory", {})
    agent_cfg = config.get("agent", {})
    ft_cfg    = config.get("fine_tuned", {})

    # ── LLM clients ────────────────────────────────────────────
    if shared_llm_client is not None:
        # Chế độ LOCAL: dùng chung 1 model đã load sẵn trong RAM
        logger.info("Using SHARED LLM client (local model in RAM — no additional load)")
        planner_llm  = shared_llm_client
        light_llm    = shared_llm_client
        reviewer_llm = shared_llm_client
        writer_llm   = shared_llm_client
    else:
        # Chế độ API (Groq/OpenAI): tạo clients riêng với temperature khác nhau
        def make_client(temperature=0.7, max_tokens=4096):
            return LLMClient(
                model_name=llm_cfg.get("model", "llama-3.3-70b-versatile"),
                backend=llm_cfg.get("backend", "openai"),
                api_base=llm_cfg.get("api_base"),
                api_key=llm_cfg.get("api_key"),
                temperature=temperature,
                max_tokens=max_tokens,
            )
        planner_llm  = make_client(temperature=llm_cfg.get("temperature", 0.7))
        light_llm    = make_client(temperature=0.3, max_tokens=1024)
        reviewer_llm = make_client(temperature=0.3)
        writer_llm   = make_client(temperature=0.5)

    # ── Fine-tuned Reranker ─────────────────────────
    reranker = None
    rk_cfg = ft_cfg.get("reranker", {})
    if rk_cfg.get("enabled", False):
        try:
            from src.models.fine_tuned_models import FineTunedReranker
            reranker = FineTunedReranker(model_path=rk_cfg["model_path"])
            logger.info(f"FineTunedReranker loaded from {rk_cfg['model_path']}")
        except Exception as e:
            logger.warning(f"FineTunedReranker load failed: {e}. Skipping reranker.")

    # ── Fine-tuned Reviewer ─────────────────────────
    fine_tuned_reviewer = None
    rv_cfg = ft_cfg.get("reviewer", {})
    if rv_cfg.get("enabled", False):
        try:
            from src.models.fine_tuned_models import FineTunedReviewer
            fine_tuned_reviewer = FineTunedReviewer(
                base_model_name=rv_cfg.get("base_model", "Qwen/Qwen2.5-14B-Instruct"),
                adapter_path=rv_cfg["adapter_path"],
            )
            logger.info(f"FineTunedReviewer loaded (adapter={rv_cfg['adapter_path']})")
        except Exception as e:
            logger.warning(f"FineTunedReviewer load failed: {e}. Falling back to API reviewer.")

    # ── Tools ───────────────────────────────────────
    search_tool = WebSearchTool(
        provider=tools_cfg.get("search_provider", "ddg"),
        max_results=tools_cfg.get("search_max_results", 5),
    )
    scraper  = WebScraper(timeout=tools_cfg.get("scrape_timeout", 15))
    memory   = EvidenceMemory(
        embedding_model=mem_cfg.get("embedding_model", "BAAI/bge-m3"),
        reranker=reranker,
    )
    tracer   = TraceLogger(trace_dir="logs/traces/ablation")
    summarizer = Summarizer(llm_client=light_llm)

    # ── Agents ──────────────────────────────────────
    planner    = Planner(llm_client=planner_llm, tracer=tracer)
    researcher = Researcher(
        llm_client=light_llm,
        search_tool=search_tool,
        scraper=scraper,
        memory=memory,
        summarizer=summarizer,
        tracer=tracer,
    )
    reviewer = Reviewer(
        llm_client=reviewer_llm,
        tracer=tracer,
        fine_tuned_reviewer=fine_tuned_reviewer,
    )
    writer = Writer(llm_client=writer_llm, tracer=tracer)

    graph = build_research_graph(
        planner=planner,
        researcher=researcher,
        reviewer=reviewer,
        writer=writer,
        max_reflection_loops=agent_cfg.get("max_reflection_loops", 3),
    )
    return graph, memory


def run_experiment(config_path: str, questions: list, output_dir: str, shared_llm_client=None) -> str:
    """Chạy 1 experiment, ghi JSONL, trả về đường dẫn file output."""
    exp_name = os.path.basename(config_path).replace(".yaml", "")
    jsonl_path = os.path.join(output_dir, f"{exp_name}_raw.jsonl")

    logger.info(f"\n{'='*60}")
    logger.info(f"EXPERIMENT: {exp_name.upper()}")
    logger.info(f"Config    : {config_path}")
    logger.info(f"Output    : {jsonl_path}")
    logger.info(f"Questions : {len(questions)}")
    logger.info(f"Mode      : {'LOCAL (shared model)' if shared_llm_client else 'API'}")
    logger.info(f"{'='*60}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Xây graph mới cho mỗi experiment (tránh shared state của agents)
    try:
        graph, memory = build_graph_from_config(config, shared_llm_client=shared_llm_client)
    except Exception as e:
        logger.error(f"Failed to build graph for {exp_name}: {e}")
        return jsonl_path

    # Xóa file cũ nếu có để ghi lại từ đầu
    if os.path.exists(jsonl_path):
        os.remove(jsonl_path)

    for idx, q in enumerate(questions):
        logger.info(f"[{exp_name}] Q{idx+1}/{len(questions)}: '{q.question[:60]}...'")

        # Reset memory session để tránh contamination giữa các câu hỏi
        memory.clear_session()

        initial_state = {
            "user_query": q.question,
            "config": config,
            "research_context": None,
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
            "reviewer_decisions": [],
        }

        final_report_text = ""
        citations_dump    = []
        evidence_dump     = []
        step_count        = 0
        success           = False

        for attempt in range(2):  # retry 1 lần nếu fail
            try:
                final_state = graph.invoke(initial_state)
                if final_state and final_state.get("final_report"):
                    final_report_text = final_state.get("final_report", "")
                    citations_dump = [
                        c.model_dump() if hasattr(c, "model_dump") else c
                        for c in final_state.get("citations", [])
                    ]
                    evidence_dump = [
                        e.model_dump() if hasattr(e, "model_dump") else e
                        for e in final_state.get("collected_evidence", [])
                    ]
                    step_count = final_state.get("step_count", 0)
                    success = True
                break
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed for {q.id}: {e}")
                if attempt == 0:
                    time.sleep(3)

        record = {
            "experiment":         exp_name,
            "question_id":        q.id,
            "question":           q.question,
            "ground_truth_answer": q.ground_truth_answer,
            "final_report":       final_report_text,
            "citations":          citations_dump,
            "collected_evidence": evidence_dump,
            "step_count":         step_count,
            "success":            success,
            "reviewer_decisions":  final_state.get("reviewer_decisions", []) if final_state else [],
        }

        # Append ngay lập tức để không mất data khi timeout
        with open(jsonl_path, "a", encoding="utf-8") as f_out:
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"  → {'OK' if success else 'FAILED'} | steps={step_count} | written to {jsonl_path}")

        # Chỉ sleep khi dùng API (tránh rate limit). Local model: không cần.
        if shared_llm_client is None:
            time.sleep(2)

    return jsonl_path


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments and save raw JSONL outputs")
    parser.add_argument("--samples",    type=int, default=5,                  help="Number of HotpotQA questions per experiment")
    parser.add_argument("--output_dir", type=str, default="data/raw_outputs", help="Directory to save JSONL files")
    parser.add_argument("--exp",        type=str, default=None,               help="Run only this experiment (e.g. exp3_agent_base). Default: all 6.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load questions
    from src.evaluation.benchmarks.loaders import HotpotQALoader
    questions = HotpotQALoader().load(n_samples=args.samples)
    logger.info(f"Loaded {len(questions)} HotpotQA questions")

    # Filter configs
    configs = DEFAULT_CONFIGS
    if args.exp:
        configs = [c for c in configs if args.exp in c]
        if not configs:
            logger.error(f"No config found matching --exp '{args.exp}'")
            sys.exit(1)

    # Run experiments
    results = {}
    for cfg_path in configs:
        if not os.path.exists(cfg_path):
            logger.warning(f"Config not found, skipping: {cfg_path}")
            continue
        jsonl_path = run_experiment(cfg_path, questions, args.output_dir)
        results[os.path.basename(cfg_path).replace(".yaml", "")] = jsonl_path

    logger.info("\n" + "="*60)
    logger.info("ALL EXPERIMENTS COMPLETE")
    for exp, path in results.items():
        lines = sum(1 for _ in open(path, encoding="utf-8")) if os.path.exists(path) else 0
        logger.info(f"  {exp}: {lines} records → {path}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
