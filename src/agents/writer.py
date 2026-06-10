"""
Writer — tổng hợp findings + evidence thành báo cáo Markdown có citation.

KHÁC SO VỚI VERSION CŨ:
- Dùng context.primary_language thay vì 'detect from user_query'
- Validate citation indices sau khi LLM sinh, flag những citation lạc
- task.findings được Writer ĐỌC (graph phải populate trước khi gọi Writer)
"""
import logging
import time
import re
from typing import List, Dict, Set
from pydantic import BaseModel, Field

from src.models.llm_server import LLMClient
from src.graph.state import (
    AgentState, WrittenReport, Citation, EvidenceModel, ResearchContext
)
from src.utils.tracer import TraceLogger

logger = logging.getLogger(__name__)


class _LLMWriterSchema(BaseModel):
    title: str = Field(description="Academic title for the report")
    report_body: str = Field(
        description="Full Markdown body with inline citations [1], [2] referencing evidence indices"
    )
    citations: List[Citation] = Field(description="Citation list, indices match those used in body")


class Writer:
    def __init__(self, llm_client: LLMClient, tracer: TraceLogger):
        self.llm_client = llm_client
        self.tracer = tracer

    # ------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------
    def write(self, state: AgentState) -> WrittenReport:
        logger.info("Writer.write() start")
        t0 = time.time()

        user_query = state["user_query"]
        context: ResearchContext = state["research_context"]
        plan = state.get("research_plan", [])
        evidence = state.get("collected_evidence", [])

        # Dedupe evidence theo URL, giữ thứ tự
        evidence_list = self._dedupe_evidence(evidence)

        findings_context = self._build_findings_context(plan, context)
        evidence_context = self._build_evidence_context(evidence_list)

        system_prompt = self._build_system_prompt(context)
        user_prompt = (
            f"Research subject: '{user_query}'\n\n"
            f"Sub-task findings (these are SYNTHESIZED facts, prioritize using them):\n"
            f"{findings_context}\n\n"
            f"Available evidence (use citation indices [i] EXACTLY matching the indices below):\n"
            f"{evidence_context}\n\n"
            f"Write the report now. Remember: language = '{context.primary_language}'."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            raw = self.llm_client.generate_structured(messages, schema=_LLMWriterSchema)
            llm_out = _LLMWriterSchema(**raw) if isinstance(raw, dict) else raw

            # Post-process: validate citations, append references
            report_body = self._validate_and_finalize_body(
                llm_out.report_body,
                llm_out.citations,
                evidence_list,
            )

            report = WrittenReport(
                title=llm_out.title,
                report_body=report_body,
                citations=llm_out.citations,
            )

            self._log(t0, user_query, report, fallback=False)
            return report

        except Exception as e:
            logger.error(f"Writer LLM failed: {e}. Using fallback layout.")
            report = self._fallback_report(user_query, plan, evidence_list, context)
            self._log(t0, user_query, report, fallback=True)
            return report

    # ------------------------------------------------------------
    # Context builders
    # ------------------------------------------------------------
    def _build_system_prompt(self, context: ResearchContext) -> str:
        return (
            "You are a senior scientific writer. Synthesize the provided research findings and "
            "evidence into a comprehensive, well-structured academic Markdown report.\n\n"
            "FORMATTING RULES:\n"
            "1. Use clear headings (##), sub-headings (###), and lists where appropriate.\n"
            "2. Support EVERY factual claim with an inline citation: [1], [2], etc.\n"
            "3. Citation indices MUST exactly match the indices of evidence provided in user message.\n"
            "4. Do NOT invent claims unsupported by the findings or evidence.\n"
            "5. Tone: objective, technical, formal.\n\n"
            "LANGUAGE RULE:\n"
            f"Write the entire report (title, body) in language code: '{context.primary_language}'. "
            "This is the user's original query language. Do NOT switch language based on the "
            "evidence's language or the geographic subject. All technical terms should be expressed "
            "correctly in the target language."
        )

    def _build_findings_context(self, plan: list, context: ResearchContext) -> str:
        """Render findings từ task.findings (đã được Reviewer populate)."""
        if not plan:
            return "(No sub-task findings recorded.)"
        parts = []
        for task in plan:
            findings = getattr(task, "findings", None) or "(no findings)"
            parts.append(
                f"### Sub-question #{task.id}: {task.sub_question}\n"
                f"Findings: {findings}"
            )
        return "\n\n".join(parts)

    def _build_evidence_context(self, evidence_list: List[EvidenceModel]) -> str:
        parts = []
        for idx, ev in enumerate(evidence_list, start=1):
            # Trim text để không nổ token
            words = ev.text.split()
            snippet = " ".join(words[:400])
            if len(words) > 400:
                snippet += "..."
            parts.append(
                f"[{idx}] Source: {ev.title} (URL: {ev.source_url})\n"
                f"Excerpt: {snippet}"
            )
        return "\n\n".join(parts)

    # ------------------------------------------------------------
    # Citation validation
    # ------------------------------------------------------------
    def _validate_and_finalize_body(
        self,
        body: str,
        citations: List[Citation],
        evidence_list: List[EvidenceModel],
    ) -> str:
        max_valid_idx = len(evidence_list)

        # Chỉ quét các dấu ngoặc vuông chứa từ 1 đến 2 chữ số (ví dụ từ [1] đến [99]) để né số năm [2026]
        used_indices = set(int(m) for m in re.findall(r"\[(\d{1,2})\]", body))
        invalid_indices = {i for i in used_indices if i < 1 or i > max_valid_idx}

        if invalid_indices:
            logger.warning(
                f"Writer used invalid citation indices: {invalid_indices} "
                f"(max valid={max_valid_idx}). Flagging."
            )
            # Đánh dấu inline, không xóa (giữ minh bạch cho debug)
            for inv in invalid_indices:
                body = re.sub(
                    rf"\[{inv}\]",
                    f"[{inv}⚠]",
                    body,
                )

        # Bảo đảm có References section
        if "## References" not in body and "## Tài liệu tham khảo" not in body:
            body += self._build_references_footer(citations, evidence_list, used_indices)

        return body

    def _build_references_footer(
        self,
        citations: List[Citation],
        evidence_list: List[EvidenceModel],
        used_indices: Set[int],
    ) -> str:
        """Build references từ citations LLM trả về. Nếu citations rỗng, dùng evidence_list."""
        lines = ["\n\n## References\n"]

        # Prefer citations LLM trả ra (có index/url/title chuẩn)
        if citations:
            sorted_cits = sorted(citations, key=lambda c: c.index)
            for c in sorted_cits:
                lines.append(f"[{c.index}] [{c.title}]({c.url})")
        else:
            # Fallback: dùng evidence_list theo index đã được cite
            for i, ev in enumerate(evidence_list, start=1):
                if i in used_indices or not used_indices:
                    lines.append(f"[{i}] [{ev.title}]({ev.source_url})")

        return "\n".join(lines)

    # ------------------------------------------------------------
    # Dedupe
    # ------------------------------------------------------------
    @staticmethod
    def _dedupe_evidence(evidence: List[EvidenceModel]) -> List[EvidenceModel]:
        seen: Set[str] = set()
        out: List[EvidenceModel] = []
        for ev in evidence:
            if ev.source_url in seen:
                continue
            seen.add(ev.source_url)
            out.append(ev)
        return out

    # ------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------
    def _fallback_report(
        self,
        user_query: str,
        plan: list,
        evidence_list: List[EvidenceModel],
        context: ResearchContext,
    ) -> WrittenReport:
        """Báo cáo tối thiểu khi LLM Writer crash. Liệt kê findings + evidence raw."""
        citations: List[Citation] = []
        body_parts = [f"# Research Report: {user_query}\n"]

        # Findings section
        if plan:
            body_parts.append("## Findings\n")
            for task in plan:
                findings = getattr(task, "findings", None) or "(no findings)"
                body_parts.append(f"### {task.sub_question}\n{findings}\n")

        # Evidence section với citation chuẩn
        body_parts.append("## Evidence Summary\n")
        for i, ev in enumerate(evidence_list[:10], start=1):
            citations.append(Citation(index=i, url=ev.source_url, title=ev.title))
            snippet = ev.text[:500] + ("..." if len(ev.text) > 500 else "")
            body_parts.append(f"**[{i}] {ev.title}**\n\n{snippet}\n")

        # References
        body_parts.append("\n## References\n")
        for c in citations:
            body_parts.append(f"[{c.index}] [{c.title}]({c.url})")

        return WrittenReport(
            title=f"Research on {user_query}",
            report_body="\n".join(body_parts),
            citations=citations,
        )

    # ------------------------------------------------------------
    # Logging helper
    # ------------------------------------------------------------
    def _log(self, t0: float, query: str, report: WrittenReport, fallback: bool) -> None:
        duration = time.time() - t0
        self.tracer.log_tool_call(
            "Writer.write",
            {"query": query, "fallback": fallback},
            f"{len(report.report_body)} chars, {len(report.citations)} citations",
            duration,
        )
        self.tracer.log_step(
            "Writer",
            "Created report" + (" (fallback)" if fallback else ""),
            query,
            {"title": report.title, "n_citations": len(report.citations)},
        )
