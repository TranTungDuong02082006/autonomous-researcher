import logging
import time
from typing import Dict, List
from pydantic import BaseModel, Field
from src.models.llm_server import LLMClient
from src.graph.state import AgentState, WrittenReport, Citation, EvidenceModel

logger = logging.getLogger(__name__)

class WrittenReportSchema(BaseModel):
    title: str = Field(description="A compelling, academic title for the final research report.")
    report_body: str = Field(description="The complete research report body formatted in beautiful Markdown. MUST include inline citation numbers like [1], [2] matching referenced links.")
    citations: List[Citation] = Field(description="A clean list of unique citations referenced in the report, sorted by index [1], [2], etc.")

class Writer:
    def __init__(self, llm_client: LLMClient, tracer: TraceLogger):
        self.llm_client = llm_client
        self.tracer = tracer

    def write(self, state: AgentState) -> WrittenReportSchema:
        """Synthesize collected task findings and evidence into a formatted research report."""
        logger.info("Writing final research report...")
        t0 = time.time()

        user_query = state.get("user_query", "")
        plan = state.get("research_plan", [])
        evidence = state.get("collected_evidence", [])

        # Create structured plan findings context
        findings_context = ""
        for task in plan:
            findings_context += f"### Sub-Question: {task.sub_question}\n"
            findings_context += f"Findings gathered: {task.findings or 'None'}\n\n"

        # Unique URLs for citation index mapping
        evidence_list = []
        seen_urls = set()
        for ev in evidence:
            if ev.source_url not in seen_urls:
                seen_urls.add(ev.source_url)
                evidence_list.append(ev)

        evidence_context = ""
        for idx, ev in enumerate(evidence_list):
            evidence_context += f"[{idx + 1}] Source: {ev.title} (URL: {ev.source_url})\nContent excerpt: {ev.text}\n\n"

        system_prompt = (
            "You are a stellar Senior Scientific Writer. Your goal is to draft a comprehensive, cohesive, "
            "and beautifully structured academic Markdown report that synthesizes all research findings. "
            "Follow these strict formatting guidelines:\n"
            "1. Organize the report with clear headings, sub-headings, lists, and tables where helpful.\n"
            "2. Support every factual claim with an inline citation in bracket format, matching the evidence index, e.g. [1], [2].\n"
            "3. Ensure the tone is objective, technical, and formal (System-2 Thinking).\n"
            "4. Return a JSON object matching the requested schema. Ensure the citations list contains all cited sources."
        )

        user_prompt = (
            f"Research Subject: '{user_query}'\n\n"
            f"Sub-Task Findings:\n"
            f"{findings_context}\n"
            f"Available Evidence References:\n"
            f"{evidence_context}\n"
            f"Draft a rigorous research report synthesizing these findings. Ensure all citation numbers in brackets [i] "
            f"refer strictly to the source indices listed in the Available Evidence References above."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            result = self.llm_client.generate_structured(messages, schema=WrittenReportSchema)
            
            # Formulate the WrittenReportSchema object
            report = WrittenReportSchema(**result)

            # Ensure we append a clean references list at the bottom of the body if not already present
            refs_str = "\n\n## References\n"
            for cit in report.citations:
                refs_str += f"[{cit.index}] [{cit.title}]({cit.url})\n"
            
            if "## References" not in report.report_body:
                report.report_body += refs_str

            duration = time.time() - t0
            self.tracer.log_tool_call("Writer.write", {"query": user_query}, f"Written report of {len(report.report_body)} chars", duration)
            self.tracer.log_step("Writer", "Created final report", user_query, report.model_dump())

            return report
        except Exception as e:
            logger.error(f"Writer generation failed: {e}. Defaulting to basic report layout.")
            # Simple fallback
            fallback_citations = []
            fallback_body = f"# Research Report: {user_query}\n\n"
            for i, ev in enumerate(evidence_list[:5]):
                fallback_citations.append(Citation(index=i+1, url=ev.source_url, title=ev.title))
                fallback_body += f"## Evidence [{i+1}]: {ev.title}\n{ev.text}\n\n"
            
            fallback_body += "\n\n## References\n"
            for cit in fallback_citations:
                fallback_body += f"[{cit.index}] [{cit.title}]({cit.url})\n"

            return WrittenReportSchema(
                title=f"Research on {user_query}",
                report_body=fallback_body,
                citations=fallback_citations
            )
