import logging
from typing import Optional
from src.models.llm_server import LLMClient

logger = logging.getLogger(__name__)

class Summarizer:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def summarize(self, text: str, query_context: str) -> str:
        """
        Compress text down to fit window limits.
        Gia cố Strict Window Clipping về 1800 từ để bảo vệ luồng chạy an toàn trước hạn mức 12k TPM.
        """
        logger.info("Summarizing text with query context...")
        
        words = text.split()
        total_words = len(words)
        
        if total_words < 300:
            return text

        # 🩹 VÁ LỖI HIỆU NĂNG: Khóa cứng cửa sổ 1800 từ để tổng tokens gửi lên luôn dưới 3000 tokens
        MAX_SAFE_WORDS = 1800
        if total_words > MAX_SAFE_WORDS:
            logger.warning(f"⚠️ Phát hiện tài liệu quá dài ({total_words} từ). Ép nén cứng về {MAX_SAFE_WORDS} từ để tránh bẫy 413 Groq...")
            words = words[:MAX_SAFE_WORDS]
            text = " ".join(words) + "\n[Truncated for context length compliance]"

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
            logger.error(f"❌ Summarization layer failed: {e}. Executing immediate substring truncation fallback.")
            return " ".join(words[:800]) + "..."
