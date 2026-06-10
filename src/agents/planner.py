"""
Planner — 2-step:
  Step 1: extract ResearchContext (locale, time, sources, geo) từ user query
  Step 2: sinh sub-tasks dựa trên context, giữ nguyên locale/time/geo anchors

KHÁC SO VỚI VERSION CŨ:
- KHÔNG còn ép "MUST English" hay "MUST Vietnamese"
- Context được extract 1 LẦN, downstream agents đọc từ đó
- Sub-question giữ nguyên geographic/temporal anchors của user
"""
import logging
import time
import datetime
from typing import List, Tuple
from pydantic import BaseModel, Field

from src.models.llm_server import LLMClient
from src.graph.state import ResearchContext, ResearchTask
from src.utils.tracer import TraceLogger

logger = logging.getLogger(__name__)


class _TasksSchema(BaseModel):
    """Schema riêng để LLM trả về list task. Tách khỏi ResearchTask để tránh
    LLM phải fill các field như `findings`/`iterations`."""
    class _TaskItem(BaseModel):
        id: int
        sub_question: str
        description: str

    plan: List[_TaskItem] = Field(description="3-7 sub-questions, viết bằng primary_language của context")


class Planner:
    def __init__(self, llm_client: LLMClient, tracer: TraceLogger):
        self.llm_client = llm_client
        self.tracer = tracer

    # ------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------
    def plan(self, query: str) -> Tuple[ResearchContext, List[ResearchTask]]:
        """
        Trả về (context, tasks).
        Caller (graph node) chịu trách nhiệm gán cả 2 vào state.
        """
        logger.info(f"Planner.plan() for query: '{query[:80]}...'")

        # STEP 1: extract context
        context = self._extract_context(query)
        logger.info(
            f"Extracted context: lang={context.primary_language}, "
            f"geo={context.geographic_scope}, "
            f"time={context.temporal_scope.description if context.temporal_scope else 'None'}, "
            f"sources={context.preferred_sources}"
        )

        # STEP 2: sinh tasks dựa trên context
        tasks = self._generate_tasks(query, context)
        logger.info(f"Generated {len(tasks)} sub-tasks")

        return context, tasks

    # ------------------------------------------------------------
    # STEP 1: extract context
    # ------------------------------------------------------------
    def _extract_context(self, query: str) -> ResearchContext:
        today = datetime.date.today().isoformat()

        system_prompt = (
            "You are a research context analyzer. Given a research query in ANY language, "
            "extract structured metadata that will guide downstream research.\n\n"
            "Today's date is " + today + ".\n\n"
            "Extraction rules:\n"
            "1. primary_language: detect language of the query text itself (ISO 639-1: 'en', 'vi', 'ja', 'ko', 'zh', 'th', 'fr', etc.).\n"
            "2. secondary_languages: list other languages useful for search. Heuristic:\n"
            "   - If query in language X targets a region whose native language is Y (X != Y), add Y.\n"
            "   - Example: English query 'gasoline prices in Vietnam' → secondary=['vi'].\n"
            "   - Example: Vietnamese query about Vietnam → secondary=[].\n"
            "   - Example: English query 'compare crypto regulation in Singapore vs Thailand' → secondary=['th','zh'].\n"
            "3. geographic_scope: country/region IF explicitly named OR strongly implied. "
            "Use null for non-geographic topics (e.g., 'how does CRISPR work').\n"
            "4. temporal_scope: extract if query mentions time.\n"
            "   - Absolute dates ('April 2026') → start_date='2026-04-01', end_date='2026-04-30'.\n"
            "   - Relative ('last 2 years' from today " + today + ") → compute absolute range.\n"
            "   - 'since 2020' → start_date='2020-01-01', end_date=today.\n"
            "   - No time mention → temporal_scope=null.\n"
            "5. preferred_sources: list ONLY domains/publications user EXPLICITLY named in the query. "
            "DO NOT invent sources. If user says 'VnExpress, Tuoi Tre, CafeF' → "
            "['vnexpress.net','tuoitre.vn','cafef.vn']. If user names no source → [].\n"
            "6. domain_field: 'finance','medical','tech','politics','science','sports','general',...\n\n"
            "Be conservative: extract ONLY what is in the query. Do not over-interpret."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analyze this query:\n'''\n{query}\n'''"},
        ]

        t0 = time.time()
        try:
            raw = self.llm_client.generate_structured(messages, schema=ResearchContext)
            context = ResearchContext(**raw) if isinstance(raw, dict) else raw
            duration = time.time() - t0
            self.tracer.log_tool_call(
                "Planner.extract_context",
                {"query": query},
                context.model_dump(),
                duration,
            )
            return context
        except Exception as e:
            logger.warning(f"Context extraction failed: {e}. Using safe defaults.")
            # Fallback an toàn: detect đại khái, để empty các field còn lại
            return ResearchContext(
                primary_language=self._guess_language(query),
                secondary_languages=[],
                geographic_scope=None,
                temporal_scope=None,
                preferred_sources=[],
                domain_field="general",
            )

    # ------------------------------------------------------------
    # STEP 2: generate tasks
    # ------------------------------------------------------------
    def _generate_tasks(self, query: str, context: ResearchContext) -> List[ResearchTask]:
        # Build context summary để LLM hiểu rõ phải giữ anchor gì
        ctx_summary_parts = [f"primary_language: {context.primary_language}"]
        if context.geographic_scope:
            ctx_summary_parts.append(f"geographic_scope: {context.geographic_scope}")
        if context.temporal_scope:
            ctx_summary_parts.append(f"time_period: {context.temporal_scope.description}")
        if context.domain_field != "general":
            ctx_summary_parts.append(f"domain: {context.domain_field}")
        ctx_summary = "\n".join(f"- {p}" for p in ctx_summary_parts)

        system_prompt = (
            "You are a top-tier research planner. Break a macro question into 3-7 "
            "sequential, concrete sub-questions (System-2 planning).\n\n"
            "MANDATORY RULES:\n"
            f"1. Write every sub-question and description in language: '{context.primary_language}' "
            "(matching the user's original query language). DO NOT translate.\n"
            "2. Every sub-question MUST preserve geographic and temporal anchors of the original query "
            "if they exist. Do not generalize them away. "
            + (
                f"This query is about '{context.geographic_scope}'"
                if context.geographic_scope else ""
            )
            + (
                f" during '{context.temporal_scope.description}'"
                if context.temporal_scope else ""
            )
            + " — every sub-question must reflect this.\n"
            "3. Assign unique ascending integer IDs starting from 1.\n"
            "4. Sub-questions should be SPECIFIC and SEARCHABLE (not abstract philosophical).\n"
            "5. Order tasks logically: context/background first, then specifics, then synthesis."
        )

        user_prompt = (
            f"User query:\n'''\n{query}\n'''\n\n"
            f"Extracted context:\n{ctx_summary}\n\n"
            f"Generate 3-7 sub-questions following the rules above."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        t0 = time.time()
        try:
            raw = self.llm_client.generate_structured(messages, schema=_TasksSchema)
            plan_items = raw.get("plan", []) if isinstance(raw, dict) else raw.plan

            tasks: List[ResearchTask] = []
            for item in plan_items:
                if isinstance(item, dict):
                    tasks.append(ResearchTask(
                        id=item["id"],
                        sub_question=item["sub_question"],
                        description=item["description"],
                        status="pending",
                    ))
                else:
                    tasks.append(ResearchTask(
                        id=item.id,
                        sub_question=item.sub_question,
                        description=item.description,
                        status="pending",
                    ))

            duration = time.time() - t0
            self.tracer.log_tool_call(
                "Planner.generate_tasks",
                {"query": query, "context_lang": context.primary_language},
                f"{len(tasks)} tasks",
                duration,
            )
            self.tracer.log_step(
                "Planner",
                "Generated research plan",
                query,
                [t.model_dump() for t in tasks],
            )
            return tasks
        except Exception as e:
            logger.error(f"Task generation failed: {e}. Fallback to single-task plan.")
            return [ResearchTask(
                id=1,
                sub_question=query,
                description="Investigate the user query directly.",
                status="pending",
            )]

    # ------------------------------------------------------------
    # Helper: language guess không cần LLM (fallback)
    # ------------------------------------------------------------
    @staticmethod
    def _guess_language(text: str) -> str:
        """Heuristic siêu đơn giản: nhìn ký tự đặc trưng. Chỉ dùng khi LLM fail."""
        # Japanese kana phải check TRƯỚC Chinese Han (vì tiếng Nhật cũng dùng Han)
        if any("\u3040" <= c <= "\u309f" or "\u30a0" <= c <= "\u30ff" for c in text):
            return "ja"
        if any("\uac00" <= c <= "\ud7af" for c in text):
            return "ko"
        if any("\u4e00" <= c <= "\u9fff" for c in text):
            return "zh"
        # Vietnamese: nhận diện bằng dấu thanh
        vn_marks = set("ăâđêôơưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")
        if any(c in vn_marks for c in text.lower()):
            return "vi"
        return "en"
