import os
import sys

# HuggingFace model cache: defaults to models/ inside project root
# Override by setting HF_HOME in your .env file
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_default_hf_home = os.path.join(_project_root, "models")
_hf_home = os.environ.get("HF_HOME", _default_hf_home)
os.environ["HF_HOME"] = _hf_home
os.environ["TRANSFORMERS_CACHE"] = _hf_home
import yaml
import gradio as gr
import json
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

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

# Setup base configurations
CONFIG_PATH = "configs/base.yaml"
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
else:
    config = {
        "llm": {"model": "Qwen/Qwen2.5-14B-Instruct", "backend": "openai"},
        "tools": {"search_provider": "ddg", "search_max_results": 5, "scrape_timeout": 15},
        "memory": {"embedding_model": "BAAI/bge-m3", "chunk_size": 512, "chunk_overlap": 50},
        "agent": {"max_reflection_loops": 3},
        "fine_tuned": {
            "reviewer": {
                "enabled": True,
                "base_model": "Qwen/Qwen2.5-14B-Instruct",
                "adapter_path": "reviewer lora"
            },
            "reranker": {
                "enabled": True,
                "model_path": "reranker model"
            }
        }
    }

# Caching model loaders (Singletons) to speed up query iterations and avoid RAM bloat
_cached_reranker = None
_cached_reviewer = None

def get_fine_tuned_reranker(model_path: str):
    global _cached_reranker
    if _cached_reranker is None:
        try:
            from src.models.fine_tuned_models import FineTunedReranker
            logger.info(f"Loading Fine-Tuned Reranker (CrossEncoder) from: {model_path} into RAM...")
            _cached_reranker = FineTunedReranker(model_path=model_path)
        except Exception as e:
            logger.error(f"Failed to load FineTunedReranker adapter: {e}. Falling back to default vector distance.")
    return _cached_reranker

def get_fine_tuned_reviewer(base_model: str, adapter_path: str):
    global _cached_reviewer
    if _cached_reviewer is None:
        try:
            from src.models.fine_tuned_models import FineTunedReviewer
            logger.info(f"Loading Fine-Tuned Reviewer LoRA adapter from: {adapter_path} (Base: {base_model}) into RAM...")
            _cached_reviewer = FineTunedReviewer(base_model_name=base_model, adapter_path=adapter_path)
        except Exception as e:
            logger.error(f"Failed to load FineTunedReviewer adapter: {e}. Falling back to standard LLMClient API serving.")
    return _cached_reviewer

def run_dual_research(query: str, max_steps: int, enable_lora_reviewer: bool, enable_reranker: bool):
    """Unified dual-stream generator returning side-by-side comparative outputs."""
    # 1. Initialize and stream Vanilla LLM response (System-1)
    yield "Stream initiating...", "Initializing System-2 Thinking components...", "{}", []

    llm_config = config.get("llm", {})
    planner_model = llm_config.get("model", "Qwen/Qwen2.5-14B-Instruct")
    
    vanilla_client = LLMClient(
        model_name=planner_model,
        backend=llm_config.get("backend"),
        quantization=llm_config.get("quantization"),
        api_base=llm_config.get("api_base"),
        api_key=llm_config.get("api_key"),
        temperature=0.7,
        max_tokens=2048
    )
    
    vanilla_text = ""
    yield "System-1 direct inference initiated...", "Initializing System-2 Thinking components...", "{}", []
    
    if hasattr(vanilla_client, "client") and hasattr(vanilla_client.client, "chat") and vanilla_client.backend in ["vllm", "openai"]:
        try:
            stream = vanilla_client.client.chat.completions.create(
                model=vanilla_client.model_name,
                messages=[{"role": "user", "content": query}],
                stream=True,
                max_tokens=2048,
                temperature=0.7
            )
            for chunk in stream:
                if len(chunk.choices) > 0:
                    content = chunk.choices[0].delta.content
                    if content:
                        vanilla_text += content
                        yield vanilla_text, "System-1 completed. Starting System-2 Research Graph...", "{}", []
        except Exception as e:
            logger.warning(f"Streaming failed for vanilla client: {e}. Falling back to standard generation.")
            vanilla_text = vanilla_client.generate([{"role": "user", "content": query}])
            yield vanilla_text, "System-1 completed. Starting System-2 Research Graph...", "{}", []
    else:
        vanilla_text = vanilla_client.generate([{"role": "user", "content": query}])
        yield vanilla_text, "System-1 completed. Starting System-2 Research Graph...", "{}", []

    # 2. Setup System-2 agents and run LangGraph Research Loop
    tracer = TraceLogger(trace_dir="logs/traces/demo")
    
    planner_llm = LLMClient(
        model_name=planner_model,
        backend=llm_config.get("backend"),
        quantization=llm_config.get("quantization"),
        api_base=llm_config.get("api_base"),
        api_key=llm_config.get("api_key"),
        temperature=llm_config.get("temperature", 0.7),
        max_tokens=llm_config.get("max_tokens", 4096)
    )

    light_model = planner_model
    light_llm = LLMClient(
        model_name=light_model,
        backend=llm_config.get("backend"),
        quantization=llm_config.get("quantization"),
        api_base=llm_config.get("api_base"),
        api_key=llm_config.get("api_key"),
        temperature=0.3,
        max_tokens=1024
    )

    reviewer_model = planner_model
    reviewer_llm = LLMClient(
        model_name=reviewer_model,
        backend=llm_config.get("backend"),
        quantization=llm_config.get("quantization"),
        api_base=llm_config.get("api_base"),
        api_key=llm_config.get("api_key"),
        temperature=0.3,
        max_tokens=4096
    )

    writer_model = planner_model
    writer_llm = LLMClient(
        model_name=writer_model,
        backend=llm_config.get("backend"),
        quantization=llm_config.get("quantization"),
        api_base=llm_config.get("api_base"),
        api_key=llm_config.get("api_key"),
        temperature=0.5,
        max_tokens=4096
    )

    tools_config = config.get("tools", {})
    search_tool = WebSearchTool(
        provider=tools_config.get("search_provider", "ddg"),
        max_results=tools_config.get("search_max_results", 5)
    )

    scraper = WebScraper(
        timeout=tools_config.get("scrape_timeout", 15)
    )

    # Dynamic Reranker Routing (Ablation state check)
    reranker = None
    if enable_reranker:
        fine_tuned_config = config.get("fine_tuned", {})
        reranker_config = fine_tuned_config.get("reranker", {})
        model_path = reranker_config.get("model_path", "reranker model")
        reranker = get_fine_tuned_reranker(model_path=model_path)
    else:
        logger.info("Fine-Tuned Reranker has been disabled by user (ablation mode).")

    memory = EvidenceMemory(
        embedding_model=config.get("memory", {}).get("embedding_model", "BAAI/bge-m3"),
        reranker=reranker
    )

    summarizer = Summarizer(llm_client=light_llm)

    planner = Planner(llm_client=planner_llm, tracer=tracer)
    
    researcher = Researcher(
        llm_client=light_llm,
        search_tool=search_tool,
        scraper=scraper,
        memory=memory,
        summarizer=summarizer,
        tracer=tracer
    )

    # Dynamic Reviewer Routing (Ablation state check)
    fine_tuned_reviewer = None
    if enable_lora_reviewer:
        fine_tuned_config = config.get("fine_tuned", {})
        reviewer_config = fine_tuned_config.get("reviewer", {})
        base_model = reviewer_config.get("base_model", "Qwen/Qwen2.5-14B-Instruct")
        adapter_path = reviewer_config.get("adapter_path", "reviewer lora")
        fine_tuned_reviewer = get_fine_tuned_reviewer(base_model=base_model, adapter_path=adapter_path)
    else:
        logger.info("Fine-Tuned Reviewer LoRA has been disabled by user (ablation mode).")

    reviewer = Reviewer(llm_client=reviewer_llm, tracer=tracer, fine_tuned_reviewer=fine_tuned_reviewer)
    writer = Writer(llm_client=writer_llm, tracer=tracer)

    # Force step count override in configuration
    config["agent"]["max_steps"] = max_steps
    
    graph = build_research_graph(
        planner=planner,
        researcher=researcher,
        reviewer=reviewer,
        writer=writer,
        max_reflection_loops=config.get("agent", {}).get("max_reflection_loops", 3)
    )

    initial_state = {
        "user_query": query,
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
        "reviewer_decisions": []
    }

    # Clear previous session memory to prevent cross-contamination between queries
    memory.clear_session()

    report = ""
    traces_json = ""
    citations_data = []

    try:
        for output in graph.stream(initial_state):
            # Parse current output node keys
            node_name = list(output.keys())[0]
            node_state = output[node_name]
            
            # Formulate progress message
            progress_msg = f"### [Active Thinking Node: {node_name.upper()}]\n"
            
            if node_name == "planner":
                plan_items = "\n".join([f"- Task {t.id}: {t.sub_question} ({t.description})" for t in node_state.get("research_plan", [])])
                progress_msg += f"Research Plan Formulated:\n{plan_items}"
            elif node_name == "researcher":
                ev_count = len(node_state.get("collected_evidence", []))
                progress_msg += f"Researcher executed sub-task. Accumulated {ev_count} evidence chunks in memory."
            elif node_name == "reviewer":
                feedback = node_state.get("review_feedback")
                if feedback:
                    progress_msg += f"Peer-Reviewer identified gaps and requested reflection/search refinement:\n\n> '{feedback}'"
                else:
                    progress_msg += "Peer-Reviewer completed sufficiency checks and approved current state."
            elif node_name == "writer":
                progress_msg += "All evidence gathered. Writer is synthesizing the final academic paper..."

            # Generate step logs in json
            steps_list = tracer.steps
            traces_json = json.dumps(steps_list, indent=2, ensure_ascii=False)
            
            # Format report and citations if writer complete
            if "final_report" in node_state and node_state["final_report"]:
                report = node_state["final_report"]
            if "citations" in node_state:
                citations_data = [[cit.index, cit.title, cit.url] for cit in node_state["citations"]]

            current_report = report if report else progress_msg
            yield vanilla_text, current_report, traces_json, citations_data

    except Exception as e:
        logger.exception("Pipeline execution crashed")
        yield vanilla_text, f"### [FATAL ERROR]\nPipeline execution crashed: {e}", "{}", []


# Building the custom styled Premium Gradio UI
custom_css = """
body {
    background-color: #0f172a !important;
    color: #f1f5f9 !important;
    font-family: 'Outfit', sans-serif !important;
}
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
    padding: 30px 20px !important;
    background: #0f172a !important;
}
.premium-card {
    background: rgba(30, 41, 59, 0.45) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    backdrop-filter: blur(16px) !important;
    border-radius: 16px !important;
    padding: 24px !important;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5) !important;
    margin-bottom: 24px !important;
}
.ablation-group {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px dashed rgba(255, 255, 255, 0.15) !important;
    border-radius: 12px !important;
    padding: 16px !important;
}
.btn-primary {
    background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 14px 0 rgba(79, 70, 229, 0.4) !important;
    transition: all 0.3s ease !important;
    border-radius: 8px !important;
}
.btn-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px 0 rgba(79, 70, 229, 0.6) !important;
}
.vanilla-header {
    border-left: 4px solid #ef4444 !important;
    padding-left: 12px !important;
    margin-bottom: 12px !important;
}
.system2-header {
    border-left: 4px solid #10b981 !important;
    padding-left: 12px !important;
    margin-bottom: 12px !important;
}
footer {
    visibility: hidden !important;
    display: none !important;
}
"""

with gr.Blocks(theme=gr.themes.Default(primary_hue="indigo", font=[gr.themes.GoogleFont("Outfit"), "sans-serif"]), css=custom_css) as demo:
    
    with gr.Row():
        gr.Markdown(
            """
            <div style="text-align: center; padding: 20px 0;">
                <h1 style="font-size: 3rem; font-weight: 800; background: linear-gradient(135deg, #6366f1 0%, #3b82f6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 5px;">🚀 Autonomous AI Researcher</h1>
                <p style="font-size: 1.2rem; color: #94a3b8; font-weight: 400;">Side-by-Side Dual-Engine Comparison: System-1 Direct vs System-2 Autonomous Research Agent</p>
            </div>
            """
        )

    with gr.Row():
        with gr.Column(scale=4, elem_classes=["premium-card"]):
            query_input = gr.Textbox(
                label="Academic / Industry Research Subject",
                placeholder="e.g. Discuss the architectural differences between GPT-4 and Claude 3.5 Sonnet",
                lines=3
            )
            
            with gr.Group(elem_classes=["ablation-group"]):
                gr.Markdown("<h3 style='margin:0 0 8px 0; color:#38bdf8;'>⚙️ Ablation Controls (Fine-Tuned Components Switch)</h3>")
                with gr.Row():
                    enable_lora_checkbox = gr.Checkbox(
                        value=True,
                        label="Enable Fine-Tuned Reviewer (LoRA Adapter)"
                    )
                    enable_reranker_checkbox = gr.Checkbox(
                        value=True,
                        label="Enable Fine-Tuned Reranker (Cross-Encoder)"
                    )
            
            with gr.Row():
                max_steps_slider = gr.Slider(
                    minimum=5,
                    maximum=30,
                    value=15,
                    step=1,
                    label="Maximum Planning Steps"
                )
                submit_btn = gr.Button("Execute Side-by-Side Comparison", variant="primary", elem_classes=["btn-primary"])

    with gr.Row():
        # LEFT COLUMN - Vanilla LLM (System-1)
        with gr.Column(scale=1, elem_classes=["premium-card"]):
            gr.Markdown(
                """
                <div class="vanilla-header">
                    <h2 style="font-size: 1.4rem; font-weight: 700; color: #ef4444; margin: 0;">🛰️ EXP1: Vanilla LLM (System-1)</h2>
                    <span style="font-size: 0.85rem; color: #ef4444; opacity: 0.85;">No search, no verification, direct prompt generation</span>
                </div>
                """
            )
            vanilla_report = gr.Markdown("Direct output from the raw base model will be streamed here.")

        # RIGHT COLUMN - Autonomous Agent (System-2)
        with gr.Column(scale=1, elem_classes=["premium-card"]):
            gr.Markdown(
                """
                <div class="system2-header">
                    <h2 style="font-size: 1.4rem; font-weight: 700; color: #10b981; margin: 0;">🧠 EXP6: Autonomous Agent Full (System-2)</h2>
                    <span style="font-size: 0.85rem; color: #10b981; opacity: 0.85;">Multi-agent planning, RAG, Web scraping, Peer-Review iterations</span>
                </div>
                """
            )
            
            with gr.Tabs():
                with gr.TabItem("📖 Báo cáo học thuật"):
                    report_output = gr.Markdown("The highly structured peer-reviewed report with external citations will be streamed here.")
                with gr.TabItem("📊 Nhật ký suy nghĩ (Traces)"):
                    trace_output = gr.Code(label="Trace Steps (JSON)", language="json", interactive=False)
                with gr.TabItem("🔗 Nguồn trích dẫn (Citations)"):
                    citations_output = gr.Dataframe(
                        headers=["Index", "Source Title", "Reference URL"],
                        datatype=["str", "str", "str"],
                        col_count=(3, "fixed")
                    )

    # Wire button trigger click
    submit_btn.click(
        fn=run_dual_research,
        inputs=[query_input, max_steps_slider, enable_lora_checkbox, enable_reranker_checkbox],
        outputs=[vanilla_report, report_output, trace_output, citations_output]
    )

if __name__ == "__main__":
    demo.launch(share=True)
