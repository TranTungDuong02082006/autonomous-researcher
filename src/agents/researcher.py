"""
Researcher — đọc ResearchContext từ state để quyết định:
  - Ngôn ngữ sinh query (primary_language + secondary_languages)
  - Domain ưu tiên (preferred_sources + locale_hints registry)
  - Date filter (temporal_scope)

KHÔNG hard-code locale, KHÔNG đoán lại context từ sub-question.
"""
import logging
import time
import re
from typing import List, Set, Optional
from pathlib import Path
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, Field

from src.models.llm_server import LLMClient
from src.tools.web_search import WebSearchTool, SearchResult
from src.tools.scraper import WebScraper
from src.memory.vector_store import EvidenceMemory
from src.tools.summarizer import Summarizer
from src.graph.state import (
    AgentState, ResearchTask, EvidenceModel, SearchQuery, ResearchContext
)
from src.utils.tracer import TraceLogger

logger = logging.getLogger(__name__)


class _QueryExpansionSchema(BaseModel):
    queries: List[str] = Field(
        description="2-3 search query variants, native language of target market"
    )


class Researcher:
    # Path tới YAML registry. Có thể override qua constructor.
    DEFAULT_LOCALE_HINTS_PATH = "configs/locale_hints.yaml"

    def __init__(
        self,
        llm_client: LLMClient,
        search_tool: WebSearchTool,
        scraper: WebScraper,
        memory: EvidenceMemory,
        summarizer: Summarizer,
        tracer: TraceLogger,
        locale_hints_path: Optional[str] = None,
        max_urls_per_task: int = 5,
        max_words_before_summarize: int = 1000,
        summarize_clip_words: int = 1800,
    ):
        self.llm_client = llm_client
        self.search_tool = search_tool
        self.scraper = scraper
        self.memory = memory
        self.summarizer = summarizer
        self.tracer = tracer
        self.max_urls_per_task = max_urls_per_task
        self.max_words_before_summarize = max_words_before_summarize
        self.summarize_clip_words = summarize_clip_words

        # Load locale hints registry
        self._locale_hints: dict = {}
        self._global_blacklist: List[str] = []
        path = Path(locale_hints_path or self.DEFAULT_LOCALE_HINTS_PATH)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            self._global_blacklist = [d.lower() for d in data.pop("global_blacklist", [])]
            self._locale_hints = {k: [d.lower() for d in v] for k, v in data.items()}
            logger.info(f"Loaded locale_hints for {list(self._locale_hints.keys())}, "
                        f"global_blacklist size={len(self._global_blacklist)}")
        else:
            logger.warning(f"locale_hints file not found: {path}. Locale boost disabled.")

    # ------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------
    def execute_task(self, task: ResearchTask, state: AgentState) -> List[EvidenceModel]:
        logger.info(f"Researcher.execute_task #{task.id}: '{task.sub_question[:80]}'")
        t0 = time.time()

        context: ResearchContext = state["research_context"]

        # 1. Sinh query variants (context-aware)
        query_variants = self._generate_queries(task, context)
        query_variants = [self._clean_query(q) for q in query_variants if self._clean_query(q)]
        if not query_variants:
            query_variants = [task.sub_question]
        logger.info(f"Query variants: {query_variants}")

        self.tracer.log_tool_call(
            "Researcher.generate_queries",
            {"task_id": task.id, "context_lang": context.primary_language},
            query_variants,
            0.0,
        )

        # 2. Search với date filter + preferred domains
        all_results = self._search_all(query_variants, context, task)

        # 3. Filter: bỏ URL đã scrape + global blacklist + boost theo locale
        filtered = self._filter_and_score(all_results, state, context)
        urls_to_scrape = [r.url for r in filtered[:self.max_urls_per_task]]
        logger.info(f"Scraping {len(urls_to_scrape)} URLs")

        # 4. Scrape + chunk + (optional) summarize + embed
        if urls_to_scrape:
            self._scrape_and_store(urls_to_scrape, task)

        # 5. Retrieve top-k bằng RAG
        retrieved = self.memory.search(task.sub_question, top_k=5)
        evidence = [
            EvidenceModel(
                text=e.text,
                source_url=e.source_url,
                title=e.title,
                score=getattr(e, "score", 0.0),
            )
            for e in retrieved
        ]

        duration = time.time() - t0
        self.tracer.log_tool_call(
            "Researcher.execute_task",
            {"task_id": task.id},
            f"{len(evidence)} evidence",
            duration,
        )
        self.tracer.log_step(
            "Researcher",
            f"Task #{task.id}",
            task.sub_question,
            [e.model_dump() for e in evidence],
        )
        return evidence

    # ------------------------------------------------------------
    # Query generation — context-aware, không hard-code
    # ------------------------------------------------------------
    def _generate_queries(self, task: ResearchTask, context: ResearchContext) -> List[str]:
        # Tập ngôn ngữ search: primary + secondary
        target_langs = [context.primary_language] + list(context.secondary_languages)
        target_langs_str = ", ".join(target_langs)

        anchor_parts = []
        if context.geographic_scope:
            anchor_parts.append(f"geographic: {context.geographic_scope}")
        if context.temporal_scope:
            anchor_parts.append(f"time: {context.temporal_scope.description}")
        anchor_str = "; ".join(anchor_parts) if anchor_parts else "no specific anchors"

        system_prompt = (
            "You are a search query engineer. Expand a research sub-question into 2-3 "
            "high-impact search queries.\n\n"
            "RULES:\n"
            f"1. Generate queries in these languages: {target_langs_str}. "
            "If multiple languages, distribute queries across them.\n"
            "2. Every query MUST preserve the geographic and temporal anchors:\n"
            f"   {anchor_str}\n"
            "3. Use keyword-style phrasing, NOT full sentences. Avoid command verbs "
            "('find', 'tell me', etc.).\n"
            "4. Diversify: each query should target a slightly different angle "
            "(facts/numbers/causes/comparison).\n"
            "5. Output exactly 2-3 queries as a JSON list of strings."
        )

        user_prompt = (
            f"Sub-question: '{task.sub_question}'\n"
            f"Task description: {task.description}\n\n"
            f"Generate 2-3 keyword queries."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            raw = self.llm_client.generate_structured(messages, schema=_QueryExpansionSchema)
            queries = raw.get("queries", []) if isinstance(raw, dict) else raw.queries
            return [q.strip() for q in queries if q and q.strip()]
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}. Using raw sub_question.")
            return [task.sub_question]

    # ------------------------------------------------------------
    # Search execution với date filter + include_domains
    # ------------------------------------------------------------
    def _search_all(
        self,
        queries: List[str],
        context: ResearchContext,
        task: ResearchTask,
    ) -> List[SearchResult]:
        # Build kwargs cho search_tool
        search_kwargs = {}
        if context.temporal_scope:
            if context.temporal_scope.start_date:
                search_kwargs["start_date"] = context.temporal_scope.start_date
            if context.temporal_scope.end_date:
                search_kwargs["end_date"] = context.temporal_scope.end_date

        all_results: List[SearchResult] = []

        # Strategy A: nếu user explicit chỉ định domain → search RIÊNG trong domain đó trước
        if context.preferred_sources:
            for q in queries:
                try:
                    res = self._safe_search(
                        q,
                        include_domains=context.preferred_sources,
                        **search_kwargs,
                    )
                    all_results.extend(res)
                except Exception as e:
                    logger.warning(f"Preferred-domain search failed '{q}': {e}")

        # Strategy B: search rộng (không restrict domain) để có thêm coverage
        for q in queries:
            try:
                res = self._safe_search(q, **search_kwargs)
                all_results.extend(res)
            except Exception as e:
                logger.warning(f"Open search failed '{q}': {e}")

        return all_results

    def _safe_search(self, query: str, **kwargs) -> List[SearchResult]:
        """Wrapper: nếu search_tool không hỗ trợ kwarg nào đó, retry bỏ kwarg đó."""
        try:
            return self.search_tool.search(query, **kwargs)
        except TypeError as e:
            # search_tool chưa hỗ trợ include_domains/date → fallback gọi không kwargs
            logger.warning(f"search_tool doesn't accept kwargs ({e}); falling back to plain search.")
            return self.search_tool.search(query)

    # ------------------------------------------------------------
    # Filter & score
    # ------------------------------------------------------------
    def _filter_and_score(
        self,
        results: List[SearchResult],
        state: AgentState,
        context: ResearchContext,
    ) -> List[SearchResult]:
        scraped_urls: Set[str] = {ev.source_url for ev in state.get("collected_evidence", [])}
        seen_urls: Set[str] = set()

        # Domains được boost: preferred_sources + locale_hints theo language
        boost_domains: Set[str] = set(d.lower() for d in context.preferred_sources)
        for lang in [context.primary_language] + list(context.secondary_languages):
            boost_domains.update(self._locale_hints.get(lang, []))

        filtered: List[SearchResult] = []
        for r in results:
            if r.url in scraped_urls or r.url in seen_urls:
                continue
            seen_urls.add(r.url)

            domain = self._extract_domain(r.url)

            # Global blacklist (Wikipedia, Britannica, …)
            if any(bd in domain for bd in self._global_blacklist):
                logger.debug(f"Blacklisted: {r.url}")
                continue

            # Boost score
            score = getattr(r, "score", 0.0) or 0.0
            if any(bd in domain for bd in boost_domains):
                score = score * 1.5 + 0.1  # additive ensure not all 0
                logger.debug(f"Boosted (locale match): {r.url}")

            # Tạo SearchResult mới với score đã tính
            try:
                r_boosted = r.model_copy(update={"score": score})  # pydantic v2
            except AttributeError:
                # Nếu SearchResult là dataclass
                from dataclasses import replace as dc_replace
                r_boosted = dc_replace(r, score=score)
            filtered.append(r_boosted)

        # Sort theo score giảm dần
        filtered.sort(key=lambda x: getattr(x, "score", 0.0) or 0.0, reverse=True)
        return filtered

    # ------------------------------------------------------------
    # Scrape + chunk + summarize + embed
    # ------------------------------------------------------------
    def _scrape_and_store(self, urls: List[str], task: ResearchTask) -> None:
        try:
            pages = self.scraper.scrape_parallel(urls)
        except Exception as e:
            logger.error(f"scrape_parallel failed entirely: {e}")
            return

        chunks_to_embed: List[str] = []
        metadata_to_embed: List[dict] = []

        for page in pages:
            text = (page.content or "").strip()
            if len(text) < 50:
                continue

            words = text.split()
            if len(words) > self.max_words_before_summarize:
                clipped = " ".join(words[:self.summarize_clip_words])
                try:
                    summary = self.summarizer.summarize(clipped, task.sub_question)
                    chunks = self._safe_chunk(summary)
                except Exception as e:
                    logger.warning(f"Summarize failed for {page.url}: {e}. Using page.chunks.")
                    chunks = list(getattr(page, "chunks", []) or [])
            else:
                chunks = list(getattr(page, "chunks", []) or [self._safe_chunk(text)[0]])

            for chunk in chunks:
                if not chunk or len(chunk.strip()) < 20:
                    continue
                chunks_to_embed.append(chunk)
                metadata_to_embed.append({
                    "source_url": page.url,
                    "title": getattr(page, "title", "") or page.url,
                })

        if chunks_to_embed:
            self.memory.add(chunks_to_embed, metadata_to_embed)
            logger.info(f"Embedded {len(chunks_to_embed)} chunks")

    def _safe_chunk(self, text: str) -> List[str]:
        """Dùng public method nếu có, fallback dùng private."""
        if hasattr(self.scraper, "chunk_text"):
            return self.scraper.chunk_text(text)
        # Fallback dùng _chunk private (best-effort)
        size = getattr(self.scraper, "chunk_size", 512)
        overlap = getattr(self.scraper, "chunk_overlap", 50)
        try:
            return self.scraper._chunk(text, size=size, overlap=overlap)
        except Exception:
            # Manual fallback
            words = text.split()
            chunks = []
            step = max(1, size - overlap)
            for i in range(0, len(words), step):
                chunks.append(" ".join(words[i:i + size]))
            return chunks

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    @staticmethod
    def _clean_query(q: str) -> str:
        if not q:
            return ""
        # Giữ Unicode (cho VN/JP/CN), chỉ bỏ ký tự đặc biệt
        q = re.sub(r"[\"'`\[\]\{\}<>]", " ", q)
        q = re.sub(r"\s+", " ", q).strip()
        return q

    @staticmethod
    def _extract_domain(url: str) -> str:
        try:
            host = urlparse(url).netloc.lower()
            if host.startswith("www."):
                host = host[4:]
            return host
        except Exception:
            return url.lower()
