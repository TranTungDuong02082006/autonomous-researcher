# 🚀 System-2 Autonomous AI Researcher

<p align="center">
  <img src="https://img.shields.io/badge/Framework-LangGraph-orange?style=for-the-badge&logo=langchain" alt="LangGraph" />
  <img src="https://img.shields.io/badge/Vector_DB-ChromaDB-blue?style=for-the-badge&logo=chromadb" alt="ChromaDB" />
  <img src="https://img.shields.io/badge/Fine--Tuning-Unsloth_LoRA-red?style=for-the-badge" alt="Unsloth LoRA" />
  <img src="https://img.shields.io/badge/UI-Gradio-ff5a00?style=for-the-badge&logo=gradio" alt="Gradio" />
</p>


An advanced, multi-agent autonomous framework designed for deep internet research, critical peer review, fact synthesis, and citation-accurate report compilation. Powered by LangGraph state machine reasoning and local vector memory.

<p align="center">
  <img src="logo_autonomous_researcher.png" alt="Autonomous Researcher Logo" width="1100" />
</p>

---

## 💡 Overview & Core Concept

**System-2 Autonomous AI Researcher** is an end-to-end framework built to fully automate the scholarly and industrial research lifecycle. Unlike standard single-shot LLM queries or naive RAG pipelines that often hallucinate or output superficial answers, this system acts as a **comprehensive, autonomous research assistant** mimicking the rigorous workflow of a human academic.

By combining structured state machines, parallel web crawling, vector-based memory retrieval, and fine-tuned critique mechanisms, the agent dynamically crawls, critiques, and synthesizes complex information into high-quality, citation-grounded reports.

### 🌟 Key Pillars of the System
- **Comprehensive Research Lifecycle Support**: The agent manages the entire process from query expansion, plan formulation, iterative search-scraping, peer-review critique, to final publication-ready synthesis.
- **System-2 Deliberation Loops**: Implements agentic self-reflection, allowing a dedicated Reviewer agent to critique draft reports and prompt the Researcher for additional information, correcting omissions and resolving contradictions before finalizing.
- **Grounded & Verifiable Citations**: Every claim in the generated report is strictly mapped back to actual source content indexed in a local vector database. The pipeline calculates citation precision to guarantee that zero hallucinated claims or links are published.
- **Continuous Optimization via Fine-Tuning**: Features integrated pipelines to extract execution traces and fine-tune specialized reviewer and reranker models, enhancing domain accuracy while lowering step count and API latency.

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

### 1. Install Dependencies
Ensure you have **Python 3.10+** installed. You can set up the environment automatically:
```bash
bash scripts/setup_env.sh
```
Or manually using pip:
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` to configure your keys and custom paths:
```bash
cp .env.example .env
```

#### Key Configurations:
- **Kaggle API Credentials (Required for fine-tuned models auto-download)**:
  To automatically download the custom fine-tuned reviewer LoRA adapter and reranker models, retrieve your Kaggle API key from [Kaggle settings](https://www.kaggle.com/settings) (click **Create New Token**), and set:
  ```env
  KAGGLE_USERNAME="your_kaggle_username"
  KAGGLE_KEY="your_kaggle_api_key"
  ```
  *If Kaggle credentials are not configured, the system will run using the base LLM without the fine-tuned components.*
  *Alternatively, you can manually download [Reviewer LoRA](https://www.kaggle.com/models/ziangtran123/reviewer-lora) and [Reranker FT](https://www.kaggle.com/models/ziangtran123/reranker-ft) and place the extracted directories `reviewer_lora/` and `reranker_ft/` directly in the project root.*

- **Search Provider (Optional)**:
  To use Tavily as the primary search engine, set:
  ```env
  TAVILY_API_KEY="your_tavily_key"
  ```
  *If left empty, search queries will automatically fall back to free, unlimited **DuckDuckGo Search**.*

- **Cloud LLM serving (Optional)**:
  By default, a local **Qwen2.5-14B** model is served out-of-the-box. To speed up execution via a cloud LLM, configure a Groq API key:
  ```env
  GROQ_API_KEY="gsk_your_key_here"
  ```

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
