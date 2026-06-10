import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    url: str
    title: str
    snippet: str
    score: float

class WebSearchTool:
    def __init__(self, provider: str = "tavily", max_results: int = 5, cache_dir: str = ".cache/search"):
        self.provider = provider.lower()
        self.max_results = max_results
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.tavily_client = None
        if self.provider == "tavily":
            api_key = os.getenv("TAVILY_API_KEY")
            if api_key:
                try:
                    from tavily import TavilyClient
                    self.tavily_client = TavilyClient(api_key=api_key)
                except ImportError:
                    logger.warning("tavily-python not installed. Falling back to DuckDuckGo.")
                    self.provider = "ddg"
            else:
                logger.warning("TAVILY_API_KEY environment variable not set. Falling back to DuckDuckGo search.")
                self.provider = "ddg"

    def search(
        self,
        query: str,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[SearchResult]:
        """Search the web for query, checking disk cache first.

        Args:
            query: The search query string.
            include_domains: Optional list of domains to restrict search to.
            exclude_domains: Optional list of domains to exclude from search.
            start_date: Optional ISO date string (YYYY-MM-DD) for filtering results after this date.
            end_date: Optional ISO date string (YYYY-MM-DD) for filtering results before this date.
        """
        # Build cache key from ALL search parameters so different filters get different cache entries
        cache_file = self._get_cache_path(query, include_domains, start_date, end_date)
        if os.path.exists(cache_file):
            logger.info(f"Search cache hit for query: '{query}'")
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                    return [SearchResult(**item) for item in cached_data]
            except Exception as e:
                logger.error(f"Failed to read search cache: {e}")

        # Perform fresh search
        logger.info(f"Performing search query: '{query}' using {self.provider}")
        results = []
        if self.provider == "tavily" and self.tavily_client:
            try:
                payload = {
                    "query": query,
                    "max_results": self.max_results,
                    "search_depth": "advanced",
                }
                if include_domains:
                    payload["include_domains"] = include_domains
                if exclude_domains:
                    payload["exclude_domains"] = exclude_domains
                if start_date:
                    payload["start_date"] = start_date
                if end_date:
                    payload["end_date"] = end_date

                response = self.tavily_client.search(**payload)
                for item in response.get("results", []):
                    results.append(SearchResult(
                        url=item.get("url", ""),
                        title=item.get("title", ""),
                        snippet=item.get("content", ""),
                        score=item.get("score", 0.0)
                    ))
            except Exception as e:
                logger.error(f"Tavily search failed: {e}. Falling back to DuckDuckGo.")
                results = self._search_ddg(query, start_date=start_date, end_date=end_date)
        else:
            results = self._search_ddg(query, start_date=start_date, end_date=end_date)

        # Deduplicate and sort by score
        deduped = self._dedupe(results)
        
        # Save to cache
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump([asdict(res) for res in deduped], f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save search cache: {e}")

        return deduped

    def _search_ddg(
        self,
        query: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[SearchResult]:
        """DuckDuckGo Search engine fallback with date filter support."""
        results = []
        try:
            from duckduckgo_search import DDGS

            # Encode date filters into query string for DDG
            search_query = query
            if start_date:
                search_query += f" after:{start_date}"
            if end_date:
                search_query += f" before:{end_date}"

            with DDGS() as ddgs:
                ddg_results = list(ddgs.text(search_query, max_results=self.max_results))
                for i, item in enumerate(ddg_results):
                    results.append(SearchResult(
                        url=item.get("href", ""),
                        title=item.get("title", ""),
                        snippet=item.get("body", ""),
                        score=1.0 - (i * 0.1) # Pseudo score based on rank
                    ))
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
        return results

    def _dedupe(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove search results with duplicate URLs or highly similar titles."""
        seen_urls = set()
        deduped = []
        for res in results:
            if not res.url or res.url in seen_urls:
                continue
            seen_urls.add(res.url)
            deduped.append(res)
        return deduped

    def _get_cache_path(
        self,
        query: str,
        include_domains: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> str:
        """Create a unique cache path based on query + all filter parameters."""
        # Include all search parameters in the hash so different filter combos get different cache entries
        cache_key_parts = [query.lower().strip()]
        if include_domains:
            cache_key_parts.append("domains:" + ",".join(sorted(include_domains)))
        if start_date:
            cache_key_parts.append(f"start:{start_date}")
        if end_date:
            cache_key_parts.append(f"end:{end_date}")
        cache_string = "|".join(cache_key_parts)
        query_hash = hashlib.md5(cache_string.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{query_hash}.json")
