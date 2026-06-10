"""
Reviewer — đánh giá xem evidence đã đủ để answer sub-question chưa.

KHÁC SO VỚI VERSION CŨ:
- Dùng context.primary_language thay vì 'detect from sub_question'
- Phân biệt rõ:
  * sufficient=True (evidence thật sự đủ)
  * sufficient=False (cần thêm evidence)
  * is_system_error=True (Reviewer crash → graph nên xử lý riêng, KHÔNG loop lại)
"""
import logging
import time
from typing import List, Optional, Any
from pydantic import BaseModel, Field

from src.models.llm_server import LLMClient
from src.graph.state import ResearchTask, EvidenceModel, ResearchContext
from src.utils.tracer import TraceLogger

logger = logging.getLogger(__name__)


class ReviewResult(BaseModel):
    """Schema output của Reviewer. Caller (graph) dùng để route."""
    sufficient: bool = Field(description="True nếu evidence đủ để answer sub-question")
    findings: str = Field(description="Tóm tắt kết luận từ evidence, viết bằng primary_language")
    missing_info: Optional[str] = Field(default=None, description="Gì còn thiếu nếu insufficient")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    is_system_error: bool = Field(default=False, description="True nếu Reviewer crash, không phải insufficient thật")
    evidence_indices_used: List[int] = Field(
        default_factory=list,
        description="Index (1-based) của evidence đã thực sự reference trong findings"
    )


class _LLMReviewSchema(BaseModel):
    """Schema gửi cho LLM — bỏ is_system_error vì LLM không cần biết."""
    sufficient: bool
    findings: str
    missing_info: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_indices_used: List[int] = Field(default_factory=list)


class Reviewer:
    def __init__(
        self,
        llm_client: LLMClient,
        tracer: TraceLogger,
        fine_tuned_reviewer: Optional[Any] = None,
        max_evidence_items: int = 5,
        max_evidence_items_finetuned: int = 5,
        max_words_per_snippet: int = 300,
    ):
        self.llm_client = llm_client
        self.tracer = tracer
        self.fine_tuned_reviewer = fine_tuned_reviewer
        self.max_evidence_items = max_evidence_items
        self.max_evidence_items_finetuned = max_evidence_items_finetuned
        self.max_words_per_snippet = max_words_per_snippet

    # ------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------
    def review(
        self,
        evidence: List[EvidenceModel],
        task: ResearchTask,
        context: ResearchContext,
    ) -> ReviewResult:
        logger.info(f"Reviewer.review task #{task.id}")
        t0 = time.time()

        # Edge case: không có evidence
        if not evidence:
            return ReviewResult(
                sufficient=False,
                findings="",
                missing_info="No evidence has been collected yet.",
                confidence=1.0,
                is_system_error=False,
            )

        # Trim evidence theo backend
        limit = (
            self.max_evidence_items_finetuned
            if self.fine_tuned_reviewer is not None
            else self.max_evidence_items
        )
        trimmed = evidence[:limit]
        evidence_context = self._build_evidence_context(trimmed)

        system_prompt = self._build_system_prompt(context)

        # Path A: fine-tuned reviewer (LoRA adapter)
        if self.fine_tuned_reviewer is not None:
            try:
                result = self._review_finetuned(task, evidence_context, system_prompt, context)
                self._log(t0, task, result, backend="finetuned")
                return result
            except Exception as e:
                logger.warning(f"Fine-tuned reviewer failed: {e}. Falling back to API.")

        # Path B: API LLM
        try:
            result = self._review_api(task, evidence_context, system_prompt, context)
            self._log(t0, task, result, backend="api")
            return result
        except Exception as e:
            logger.error(f"API reviewer crashed: {e}")
            # ❗ SAFE FAILURE: KHÔNG giả vờ sufficient=True
            return ReviewResult(
                sufficient=False,
                findings="",
                missing_info=f"[SYSTEM_ERROR] Reviewer failed: {str(e)[:200]}",
                confidence=0.0,
                is_system_error=True,
            )

    # ------------------------------------------------------------
    # Prompt building (context-aware)
    # ------------------------------------------------------------
    def _build_system_prompt(self, context: ResearchContext) -> str:
        return (
            "You are a rigorous, skeptical academic peer reviewer. Analyze the provided web "
            "evidence and determine whether it contains sufficient, credible facts to FULLY answer "
            "a specific research sub-question.\n\n"
            "REVIEW RULES:\n"
            "1. Be critical. If evidence is generic, off-topic, or lacks substance → sufficient=false.\n"
            "2. List specifically what crucial facts are missing in missing_info.\n"
            "3. Cite evidence by 1-based index in evidence_indices_used (e.g., [1,3] if you used "
            "evidence #1 and #3).\n"
            "4. Confidence: how sure you are about your sufficiency judgment (0.0-1.0).\n\n"
            "LANGUAGE RULE:\n"
            f"Write 'findings' and 'missing_info' in language code: '{context.primary_language}'. "
            "This is the user's original query language. Do NOT mirror the language of the "
            "scraped evidence (which may differ from the user's language). Do NOT mix languages."
        )

    def _build_evidence_context(self, evidence: List[EvidenceModel]) -> str:
        parts = []
        for i, ev in enumerate(evidence, start=1):
            words = ev.text.split()
            snippet = " ".join(words[:self.max_words_per_snippet])
            if len(words) > self.max_words_per_snippet:
                snippet += "..."
            parts.append(
                f"[{i}] Source: {ev.title} ({ev.source_url})\n"
                f"Content excerpt: {snippet}"
            )
        return "\n\n".join(parts)

    # ------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------
    def _review_api(
        self,
        task: ResearchTask,
        evidence_context: str,
        system_prompt: str,
        context: ResearchContext,
    ) -> ReviewResult:
        user_prompt = (
            f"Sub-question: '{task.sub_question}'\n"
            f"Task description: {task.description}\n\n"
            f"Collected evidence:\n{evidence_context}\n\n"
            f"Analyze sufficiency and provide findings. "
            f"Remember: write findings/missing_info in '{context.primary_language}'."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        raw = self.llm_client.generate_structured(messages, schema=_LLMReviewSchema)
        if isinstance(raw, dict):
            llm_out = _LLMReviewSchema(**raw)
        else:
            llm_out = raw

        return ReviewResult(
            sufficient=llm_out.sufficient,
            findings=llm_out.findings,
            missing_info=llm_out.missing_info if not llm_out.sufficient else None,
            confidence=llm_out.confidence,
            is_system_error=False,
            evidence_indices_used=llm_out.evidence_indices_used,
        )

    def _review_finetuned(
        self,
        task: ResearchTask,
        evidence_context: str,
        system_prompt: str,
        context: ResearchContext,
    ) -> ReviewResult:
        """Gọi LoRA-tuned model. Adapter expose .review(claim, evidence, system_prompt)."""
        out = self.fine_tuned_reviewer.review(
            claim=task.sub_question,
            evidence=evidence_context,
            system_prompt=system_prompt,
        )
        return ReviewResult(
            sufficient=getattr(out, "sufficient", False),
            findings=getattr(out, "findings", ""),
            missing_info=getattr(out, "missing_info", None) if not getattr(out, "sufficient", False) else None,
            confidence=getattr(out, "confidence", 0.5),
            is_system_error=False,
            evidence_indices_used=getattr(out, "evidence_indices_used", []),
        )

    # ------------------------------------------------------------
    # Logging helper
    # ------------------------------------------------------------
    def _log(self, t0: float, task: ResearchTask, result: ReviewResult, backend: str) -> None:
        duration = time.time() - t0
        self.tracer.log_tool_call(
            "Reviewer.review",
            {"task_id": task.id, "backend": backend},
            {
                "sufficient": result.sufficient,
                "confidence": result.confidence,
                "is_system_error": result.is_system_error,
            },
            duration,
        )
        self.tracer.log_step(
            "Reviewer",
            f"task #{task.id} ({backend})",
            task.sub_question,
            result.model_dump(),
        )
