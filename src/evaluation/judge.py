import logging
import time
from typing import Dict
from pydantic import BaseModel, Field
from src.models.llm_server import LLMClient

logger = logging.getLogger(__name__)

class JudgeResultSchema(BaseModel):
    comprehensiveness: float = Field(description="Score from 1.0 to 5.0 of how thoroughly the report answers all facets of the target question.")
    logic_and_structure: float = Field(description="Score from 1.0 to 5.0 of the logical layout, coherence, and professional formatting of the text.")
    depth_of_research: float = Field(description="Score from 1.0 to 5.0 of factual depth, avoidance of generic statements, and inclusion of concrete details.")
    overall_score: float = Field(description="The average of the component scores, from 1.0 to 5.0.")
    justification: str = Field(description="A brief explanation for the assigned scores and feedback on improvement.")

class LLMJudge:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def judge_report(self, report_text: str, question: str, rubric: str) -> JudgeResultSchema:
        """Call LLM client to score a generated research report based on a detailed rubric."""
        logger.info("LLMJudge scoring report...")
        
        system_prompt = (
            "You are a strict, objective Academic Board Judge. Your role is to rate scientific reports "
            "strictly according to the provided rubric and score them from 1.0 (poor) to 5.0 (perfect) "
            "along with a critical justification for your score. Be highly analytical and do not give perfect scores easily."
        )

        user_prompt = (
            f"Target Question: '{question}'\n\n"
            f"Evaluation Rubric:\n"
            f"{rubric}\n\n"
            f"--- GENERATED REPORT START ---\n"
            f"{report_text}\n"
            f"--- GENERATED REPORT END ---\n\n"
            f"Evaluate the generated report and return a structured JudgeResult JSON object."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            raw_res = self.llm_client.generate_structured(messages, schema=JudgeResultSchema)
            return JudgeResultSchema(**raw_res)
        except Exception as e:
            logger.error(f"LLM Judge scoring failed: {e}. Returning default mid-tier scores.")
            return JudgeResultSchema(
                comprehensiveness=3.0,
                logic_and_structure=3.0,
                depth_of_research=3.0,
                overall_score=3.0,
                justification=f"Failed to score report due to error: {e}"
            )

    def judge_citation(self, claim: str, evidence_text: str) -> bool:
        """Verify if a specific factual claim is fully supported by the referenced evidence chunk."""
        system_prompt = (
            "You are a factual verification assistant. Determine whether the provided claim is directly "
            "supported, contradicted, or not mentioned by the evidence. Respond only with a JSON object "
            "containing a single boolean flag 'supported'."
        )

        user_prompt = (
            f"Factual Claim: '{claim}'\n\n"
            f"Source Evidence: '{evidence_text}'\n\n"
            f"Is the claim fully supported by the evidence?"
        )

        class CitationSupport(BaseModel):
            supported: bool

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            res = self.llm_client.generate_structured(messages, schema=CitationSupport)
            return res.get("supported", False)
        except Exception:
            return False
