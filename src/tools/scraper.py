import logging
import concurrent.futures
from dataclasses import dataclass, field
from typing import List, Optional
import trafilatura

logger = logging.getLogger(__name__)

@dataclass
class ScrapedPage:
    url: str
    title: str
    content: str
    chunks: List[str] = field(default_factory=list)

class WebScraper:
    def __init__(self, timeout: int = 15, chunk_size: int = 512, chunk_overlap: int = 50):
        self.timeout = timeout
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def scrape(self, url: str) -> Optional[ScrapedPage]:
        """Scrape raw page content using trafilatura and split into chunks."""
        logger.info(f"Scraping URL: {url}")
        try:
            # Fetch HTML with timeout
            downloaded = trafilatura.fetch_url(url, no_ssl=True)
            if not downloaded:
                logger.warning(f"Failed to fetch content from URL: {url}")
                return None

            # Extract main content and title
            content = trafilatura.extract(
                downloaded, 
                include_comments=False, 
                include_tables=True,
                no_fallback=False
            )
            
            if not content:
                logger.warning(f"Could not extract clean text from URL: {url}")
                return None

            # Attempt to extract title
            metadata = trafilatura.extract_metadata(downloaded)
            title = metadata.title if metadata and metadata.title else "Untitled Page"

            # Filter junk and chunk content
            clean_content = self._filter_junk(content)
            chunks = self._chunk(clean_content, size=self.chunk_size, overlap=self.chunk_overlap)

            return ScrapedPage(
                url=url,
                title=title,
                content=clean_content,
                chunks=chunks
            )
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return None

    def scrape_parallel(self, urls: List[str]) -> List[ScrapedPage]:
        """Scrape multiple URLs in parallel using ThreadPoolExecutor."""
        pages = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_url = {executor.submit(self.scrape, url): url for url in urls}
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    page = future.result()
                    if page:
                        pages.append(page)
                except Exception as exc:
                    logger.error(f"{url} generated an exception: {exc}")
        return pages

    def _filter_junk(self, text: str) -> str:
        """Remove remaining boilerplate footer/header strings or ads."""
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            line_strip = line.strip()
            # Skip very short lines that look like buttons or menu labels
            if len(line_strip) == 0:
                continue
            if line_strip.lower() in ["menu", "navigation", "search", "share", "follow us", "cookie policy", "privacy policy"]:
                continue
            cleaned_lines.append(line_strip)
        return "\n".join(cleaned_lines)

    def _chunk(self, text: str, size: int = 512, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks of words."""
        words = text.split()
        chunks = []
        
        if len(words) <= size:
            return [text]

        i = 0
        while i < len(words):
            chunk_words = words[i:i + size]
            chunk_text = " ".join(chunk_words)
            chunks.append(chunk_text)
            i += (size - overlap)
            
        return chunks
