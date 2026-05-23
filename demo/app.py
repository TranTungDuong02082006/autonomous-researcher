import os
import sys
import yaml
import gradio as gr
import json

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
        "memory": {"embedding_model": "all-MiniLM-L6-v2", "chunk_size": 512, "chunk_overlap": 50},
        "agent": {"max_reflection_loops": 3}
    }

def run_research(query: str, max_steps: int):
    """Gradio generator yielding intermediate outputs representing agent steps in real-time."""
    yield "Initializing System-2 Thinking components...", "{}", []

    tracer = TraceLogger(trace_dir="logs/traces/demo")
    
    llm_config = config.get("llm", {})
    llm_client = LLMClient(
        model_name=llm_config.get("model"),
        backend=llm_config.get("backend"),
        quantization=llm_config.get("quantization"),
        api_base=llm_config.get("api_base"),
        api_key=llm_config.get("api_key"),
        max_tokens=4096
    )

    tools_config = config.get("tools", {})
    search_tool = WebSearchTool(
        provider=tools_config.get("search_provider", "ddg"), # Default to ddg for demo ease
        max_results=tools_config.get("search_max_results", 5)
    )

    scraper = WebScraper(
        timeout=tools_config.get("scrape_timeout", 15)
    )

    memory = EvidenceMemory(
        embedding_model=config.get("memory", {}).get("embedding_model", "all-MiniLM-L6-v2")
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

    report = ""
    traces_json = ""
    citations_data = []

    # Stream graph chunks state changes
    try:
        for output in graph.stream(initial_state):
            # Parse current output node keys
            node_name = list(output.keys())[0]
            node_state = output[node_name]
            
            # Formulate progress message
            progress_msg = f"### [Active Node: {node_name.upper()}]\n"
            
            if node_name == "planner":
                plan_items = "\n".join([f"- Task {t.id}: {t.sub_question} ({t.description})" for t in node_state.get("research_plan", [])])
                progress_msg += f"Research Plan Formulated:\n{plan_items}"
            elif node_name == "researcher":
                ev_count = len(node_state.get("collected_evidence", []))
                progress_msg += f"Researcher executed task. Accumulated {ev_count} evidence chunks."
            elif node_name == "reviewer":
                feedback = node_state.get("review_feedback")
                if feedback:
                    progress_msg += f"Peer-Reviewer requested changes: '{feedback}'"
                else:
                    progress_msg += "Peer-Reviewer approved current evidence task."
            elif node_name == "writer":
                progress_msg += "Synthesizing final research paper..."

            # Generate step logs in json
            steps_list = tracer.steps
            traces_json = json.dumps(steps_list, indent=2, ensure_ascii=False)
            
            # Format report and citations if writer complete
            if "final_report" in node_state:
                report = node_state["final_report"]
            if "citations" in node_state:
                citations_data = [[cit.index, cit.title, cit.url] for cit in node_state["citations"]]

            current_report = report if report else progress_msg
            yield current_report, traces_json, citations_data

    except Exception as e:
        yield f"### [FATAL ERROR]\nPipeline execution crashed: {e}", "{}", []


# Building the custom styled Premium Gradio UI
with gr.Blocks(theme=gr.themes.Default(primary_hue="indigo", font=[gr.themes.GoogleFont("Outfit"), "sans-serif"]), css="footer {visibility: hidden}") as demo:
    
    with gr.Row():
        gr.Markdown(
            """
            <div style="text-align: center; padding: 20px 0;">
                <h1 style="font-size: 2.8rem; font-weight: 800; color: #3b82f6; margin-bottom: 5px;">🚀 Autonomous AI Researcher</h1>
                <p style="font-size: 1.15rem; color: #64748b;">System-2 Multi-Agent Research, Verification, and Fact Synthesis</p>
            </div>
            """
        )

    with gr.Row():
        with gr.Column(scale=1):
            query_input = gr.Textbox(
                label="Academic / Industry Research Subject",
                placeholder="e.g. Discuss the architectural differences between GPT-4 and Claude 3.5 Sonnet",
                lines=3
            )
            max_steps_slider = gr.Slider(
                minimum=5,
                maximum=30,
                value=15,
                step=1,
                label="Maximum Planning Steps"
            )
            submit_btn = gr.Button("Initialize Autonomous Research", variant="primary")
            
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("📖 Academic Report"):
                    report_output = gr.Markdown("Enter a query and run the research to generate a comprehensive peer-reviewed report.")
                with gr.TabItem("📊 Execution Logs & Traces"):
                    trace_output = gr.Code(label="Trace Steps (JSON)", language="json", interactive=False)
                with gr.TabItem("🔗 Citation Sources"):
                    citations_output = gr.Dataframe(
                        headers=["Index", "Source Title", "Reference URL"],
                        datatype=["str", "str", "str"],
                        col_count=(3, "fixed")
                    )

    # Wire button trigger click
    submit_btn.click(
        fn=run_research,
        inputs=[query_input, max_steps_slider],
        outputs=[report_output, trace_output, citations_output]
    )

if __name__ == "__main__":
    demo.launch(share=True)
