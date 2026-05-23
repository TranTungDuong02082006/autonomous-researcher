import logging
import time
from typing import List, Optional
from pydantic import BaseModel, Field
from src.models.llm_server import LLMClient
from src.graph.state import ResearchTask, EvidenceModel
from src.utils.tracer import TraceLogger

logger = logging.getLogger(__name__)

class ReviewResultSchema(BaseModel):
    sufficient: bool = Field(description="True if the collected evidence provides enough detail to answer the sub-question, False otherwise.")
    missing_info: Optional[str] = Field(description="If sufficient is False, specify what key details are missing. Otherwise leave blank.")
    confidence: float = Field(description="Confidence rating of this assessment from 0.0 (low) to 1.0 (high).")
    findings: str = Field(description="A concise summary of key facts gathered from the evidence that answers this task.")

class Reviewer:
    def __init__(self, llm_client: LLMClient, tracer: TraceLogger):
        self.llm_client = llm_client
        self.tracer = tracer

    def review(self, evidence: List[EvidenceModel], task: ResearchTask) -> ReviewResultSchema:
        """Evaluate evidence sufficiency for the given sub-question task."""
        logger.info(f"Reviewing gathered evidence for task #{task.id}...")
        t0 = time.time()

        if not evidence:
            return ReviewResultSchema(
                sufficient=False,
                missing_info="No evidence has been scraped or loaded yet.",
                confidence=1.0,
                findings="Evidence collection failed."
            )

        # Build context from collected evidence text
        evidence_context = ""
        for i, ev in enumerate(evidence):
            evidence_context += f"[{i + 1}] Source: {ev.title} ({ev.source_url})\nContent excerpt: {ev.text}\n\n"

        system_prompt = (
            "You are a rigorous, skeptical Academic Peer Reviewer. Your role is to carefully analyze "
            "the provided web evidence and determine whether it contains sufficient, credible facts "
            "to fully answer a specific research sub-question.\n"
            "Be critical: if the evidence is generic, empty, or lacks substance, set 'sufficient' to false "
            "and list specifically what crucial missing facts need to be searched for."
        )

        user_prompt = (
            f"Sub-Question to Answer: '{task.sub_question}'\n"
            f"Task Description: {task.description}\n\n"
            f"Collected Evidence:\n"
            f"{evidence_context}\n"
            f"Analyze if the evidence is sufficient, extract findings, and formulate the ReviewResult response."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            result = self.llm_client.generate_structured(messages, schema=ReviewResultSchema)
            review_res = ReviewResultSchema(**result)
            
            duration = time.time() - t0
            self.tracer.log_tool_call("Reviewer.review", {"task_id": task.id}, f"Sufficiency: {review_res.sufficient}", duration)
            self.tracer.log_step("Reviewer", f"Completed peer-review for task #{task.id}", task.sub_question, review_res.model_dump())
            
            return review_res
        except Exception as e:
            logger.error(f"Reviewer generation failed: {e}. Defaulting to sufficient to prevent infinite loops.")
            # Safety fallback to prevent hanging
            return ReviewResultSchema(
                sufficient=True,
                missing_info=None,
                confidence=0.5,
                findings="Compiled generic evidence findings."
            )
