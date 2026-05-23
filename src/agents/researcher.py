import logging
import time
from typing import List, Set
from pydantic import BaseModel, Field
from src.models.llm_server import LLMClient
from src.tools.web_search import WebSearchTool, SearchResult
from src.tools.scraper import WebScraper
from src.memory.vector_store import EvidenceMemory
from src.tools.summarizer import Summarizer
from src.graph.state import AgentState, ResearchTask, EvidenceModel, SearchQuery
from src.utils.tracer import TraceLogger

logger = logging.getLogger(__name__)

class QueryExpansionSchema(BaseModel):
    queries: List[str] = Field(description="A list of exactly 2 to 3 diverse search query variants targeting the sub-question.")

class Researcher:
    def __init__(
        self,
        llm_client: LLMClient,
        search_tool: WebSearchTool,
        scraper: WebScraper,
        memory: EvidenceMemory,
        summarizer: Summarizer,
        tracer: TraceLogger
    ):
        self.llm_client = llm_client
        self.search_tool = search_tool
        self.scraper = scraper
        self.memory = memory
        self.summarizer = summarizer
        self.tracer = tracer

    def execute_task(self, task: ResearchTask, state: AgentState) -> List[EvidenceModel]:
        """Perform search query expansion, scrape new URLs, embed to memory, and retrieve top evidence."""
        logger.info(f"Executing Researcher task #{task.id}: '{task.sub_question}'")
        t0 = time.time()

        # 1. Generate search query variants
        query_variants = self._generate_queries(task)
        logger.info(f"Expanded queries: {query_variants}")

        # Record queries in search history
        new_queries = [SearchQuery(query=q, timestamp=time.strftime("%Y-%m-%d %H:%M:%S")) for q in query_variants]

        # 2. Execute searches and deduplicate URLs
        all_results = []
        for q in query_variants:
            search_res = self.search_tool.search(q)
            all_results.extend(search_res)

        # Filter out redundant search results (already scraped or already processed)
        filtered_results = self._filter_redundant(all_results, state)
        urls_to_scrape = [res.url for res in filtered_results][:5] # Scrape top 5 new URLs maximum per step

        # 3. Process new pages: parallel scrape, chunk, summarize (if long), and index to vector store
        if urls_to_scrape:
            logger.info(f"Scraping {len(urls_to_scrape)} new URLs in parallel: {urls_to_scrape}")
            scraped_pages = self.scraper.scrape_parallel(urls_to_scrape)
            
            chunks_to_embed = []
            metadata_to_embed = []
            
            for page in scraped_pages:
                # If page body is very long, summarize it to fit window limits
                full_text = page.content
                if len(full_text.split()) > 1000:
                    summary = self.summarizer.summarize(full_text, task.sub_question)
                    # Re-chunk the summary
                    chunks = self.scraper._chunk(summary, size=self.scraper.chunk_size, overlap=self.scraper.chunk_overlap)
                else:
                    chunks = page.chunks

                for chunk in chunks:
                    chunks_to_embed.append(chunk)
                    metadata_to_embed.append({
                        "source_url": page.url,
                        "title": page.title
                    })

            # Add new chunks to session memory
            if chunks_to_embed:
                self.memory.add(chunks_to_embed, metadata_to_embed)

        # 4. Query vector store to retrieve top relevant Evidence for this specific sub-question
        retrieved_evidence = self.memory.search(task.sub_question, top_k=5)
        
        # Convert memory Evidence dataclass to state EvidenceModel Pydantic model
        state_evidence = []
        for ev in retrieved_evidence:
            state_evidence.append(EvidenceModel(
                text=ev.text,
                source_url=ev.source_url,
                title=ev.title,
                score=ev.score
            ))

        duration = time.time() - t0
        self.tracer.log_tool_call(
            "Researcher.execute_task", 
            {"task_id": task.id, "sub_question": task.sub_question}, 
            f"Collected {len(state_evidence)} evidence documents.", 
            duration
        )
        self.tracer.log_step(
            "Researcher", 
            f"Executed research for task #{task.id}", 
            task.sub_question, 
            [ev.model_dump() for ev in state_evidence]
        )

        return state_evidence

    def _generate_queries(self, task: ResearchTask) -> List[str]:
        """Generate 2-3 search query variations from sub-question using LLM."""
        system_prompt = (
            "You are an expert search engine engineer. Your job is to take a research sub-question "
            "and expand it into 2 to 3 high-impact, diverse keyword search queries. "
            "Generate distinct angles (e.g. key architectures, definitions, recent statistics). "
            "Return a clean JSON object containing the queries list."
        )
        user_prompt = f"Generate 2 to 3 search query variants for: '{task.sub_question}'"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            expanded = self.llm_client.generate_structured(messages, schema=QueryExpansionSchema)
            queries = expanded.get("queries", [])
            # Fallback if empty
            if not queries:
                queries = [task.sub_question]
            return queries
        except Exception:
            return [task.sub_question]

    def _filter_redundant(self, results: List[SearchResult], state: AgentState) -> List[SearchResult]:
        """Filter out search results that are already scraped or recorded in gathered evidence."""
        # Find URLs we've already collected
        scraped_urls: Set[str] = {ev.source_url for ev in state.get("collected_evidence", [])}
        
        filtered = []
        for res in results:
            if res.url not in scraped_urls:
                filtered.append(res)
        return filtered
