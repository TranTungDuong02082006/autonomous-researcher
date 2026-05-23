# 🚀 System-2 Autonomous AI Researcher

An advanced, multi-agent autonomous framework designed for deep internet research, critical peer review, fact synthesis, and citation-accurate report compilation. Powered by LangGraph state machine reasoning and local vector memory.

---

## 📂 Repository Structure

```text
autonomous-researcher/
├── configs/                 # YAML configuration environments
│   └── experiments/         # Ablation and comparative configs
├── src/
│   ├── agents/             # System-2 reasoning agents (Planner, Researcher, Reviewer, Writer)
│   ├── tools/              # Scraper (Trafilatura), Search (Tavily/DDG), Summarizer
│   ├── models/             # LLM API wrappers and fine-tuned model loaders
│   ├── memory/             # Local Vector DB memory (ChromaDB + SentenceTransformers)
│   ├── graph/              # LangGraph compilation layers & edge logic
│   ├── training/           # Unsloth fine-tuning datasets and LoRA training scripts
│   ├── evaluation/         # Metrics (F1, citations), Judges, and parallel Runner
│   └── utils/              # Tracer loggers and helper modules
├── scripts/                # CLI entry points
├── tests/                  # Pytest unit tests suites
├── demo/                   # Gradio premium web dashboard
├── requirements.txt        # Package dependencies specification
└── README.md               # Overview documentation
```

---

## ⚙️ Environment Setup

### Install Dependencies
Ensure you have Python 3.10+ installed. Execute the setup script or install via pip:
```bash
bash scripts/setup_env.sh
```

Or manually:
```bash
pip install -r requirements.txt
```

### Search API Key Configuration (Optional)
To use Tavily as your primary search tool, set the environment variable:
```bash
export TAVILY_API_KEY="your-tavily-api-key"
```
*If not provided, the search provider will automatically fall back to completely free, unlimited **DuckDuckGo Search**.*

---

## ⚡ Quick Start

### 1. Test LLM Serving client
You can test the client wrapper by executing:
```bash
python scripts/test_llm.py "What is LangGraph?"
```
*Tip: Update `backend` inside `configs/base.yaml` to `"openai"` or `"transformers"` based on your serving setup.*

### 2. Trigger Autonomous Research Agent
Execute a macro research request directly from the CLI:
```bash
python scripts/run_agent.py --query "Discuss the context window length limits of Claude 3.5 Sonnet vs Llama 3 70B"
```
The agent will formulate a plan, run search query expansions, scrape pages in parallel, critique outcomes, synthesize findings, and write a formatted peer-reviewed document to `output_report.md`.

---

## 📖 Evaluation & Replication

### Parallel Benchmark Running
To run QA evaluation over HotpotQA dev validation questions in parallel:
```bash
python scripts/run_benchmark.py --benchmark hotpotqa --samples 3
```

### Ablation Experiments Batch Run
To run and compare vanilla direct-prompting vs system-2 planning agentic pipelines:
```bash
python scripts/run_all_experiments.py --benchmark hotpotqa --samples 2
```
Aggregated metrics comparison matrices will be compiled to `logs/experiments/experiments_comparison.csv`.

---

## 🔮 Fine-Tuning Pipeline (Phase 5)

We provide tools to build synthetic instruction datasets from agent tracing files and trigger fast parameter-efficient SFT fine-tuning using `Unsloth` (for fast GPU adapters training):

1. **Build synthetic review datasets**:
   ```bash
   python -c "from src.training.data.build_datasets import ReviewerDatasetBuilder; builder=ReviewerDatasetBuilder(); builder.generate_synthetic_from_traces([])"
   ```
2. **LoRA Fine-tuning Training**:
   We support fast training on Blackwell/Ada GPUs:
   ```bash
   python -c "from src.training.trainers import train_reviewer; train_reviewer('configs/base.yaml')"
   ```

---

## 🖥️ Premium Gradio Demo Dashboard

Launch the side-by-side interactive dashboard to observe planning steps, JSON traces, and citations live:
```bash
python demo/app.py
```
This will launch a local server and provide a public shareable URL.
