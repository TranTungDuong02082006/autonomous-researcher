import logging
from typing import Optional
from src.models.llm_server import LLMClient

logger = logging.getLogger(__name__)

class Summarizer:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def summarize(self, text: str, query_context: str) -> str:
        """
        Compress text (e.g. 5-10k tokens) down to 500-1000 tokens focusing on query_context.
        Mitigates context window issues.
        """
        logger.info("Summarizing text with query context...")
        
        # If the text is very short, no need to summarize
        if len(text.split()) < 300:
            return text

        system_prompt = (
            "You are an expert academic summarizer. Your job is to compress a long source text "
            "into a highly information-dense, accurate summary. Keep only relevant facts, figures, "
            "and references, and completely discard navbars, advertisements, or boilerplate junk."
        )

        user_prompt = (
            f"Please summarize the following text, focusing particularly on information relevant to the query context: '{query_context}'.\n\n"
            f"--- SOURCE TEXT START ---\n"
            f"{text}\n"
            f"--- SOURCE TEXT END ---\n\n"
            f"Your output summary should be highly concise (between 300 to 800 words), well-structured with bullet points if necessary, "
            f"and contain direct factual claims without fluff."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            summary = self.llm_client.generate(messages, temperature=0.3, max_tokens=1000)
            return summary.strip()
        except Exception as e:
            logger.error(f"Summarization failed: {e}. Returning truncated text as fallback.")
            # Fallback to simple truncation
            words = text.split()
            return " ".join(words[:800]) + "..."
