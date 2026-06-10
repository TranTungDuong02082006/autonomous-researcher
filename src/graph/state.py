from typing import Dict, List, Literal, Optional, TypedDict, Any, Annotated
import operator
from pydantic import BaseModel, Field


# ============================================================
# RESEARCH CONTEXT — extract 1 lần ở Planner, dùng xuyên suốt
# ============================================================

class TemporalScope(BaseModel):
    """Phạm vi thời gian nếu user query có đề cập."""
    start_date: Optional[str] = Field(
        default=None,
        description="ISO format YYYY-MM-DD. Null nếu không xác định."
    )
    end_date: Optional[str] = Field(
        default=None,
        description="ISO format YYYY-MM-DD. Null nếu không xác định."
    )
    description: str = Field(
        description="Mô tả thời gian theo cách user viết, ví dụ 'April 2026', 'last 2 years'."
    )


class ResearchContext(BaseModel):
    """
    Metadata vĩ mô được extract 1 LẦN từ user query bởi Planner.
    Tất cả downstream agents (Researcher/Reviewer/Writer) đọc từ object này,
    KHÔNG đoán lại từ sub-question.
    """
    primary_language: str = Field(
        description="ISO 639-1 code của ngôn ngữ user query. Ví dụ: 'en', 'vi', 'ja'."
    )
    secondary_languages: List[str] = Field(
        default_factory=list,
        description=(
            "Các ngôn ngữ khác có liên quan để mở rộng search. "
            "Ví dụ: query English về Vietnam → ['vi']."
        )
    )
    geographic_scope: Optional[str] = Field(
        default=None,
        description=(
            "Country/region focus nếu có. Ví dụ 'Vietnam', 'Southeast Asia', 'global'. "
            "Null nếu topic không có anchor địa lý (vd 'how does CRISPR work')."
        )
    )
    temporal_scope: Optional[TemporalScope] = Field(
        default=None,
        description="Phạm vi thời gian nếu user query có đề cập."
    )
    preferred_sources: List[str] = Field(
        default_factory=list,
        description=(
            "Domain user EXPLICITLY mention trong query. Ví dụ: query nói "
            "'from VnExpress, Tuoi Tre' → ['vnexpress.net', 'tuoitre.vn']. "
            "KHÔNG được tự bịa source."
        )
    )
    domain_field: str = Field(
        default="general",
        description="Topic field: 'finance', 'medical', 'tech', 'politics', 'general', etc."
    )


# ============================================================
# RESEARCH TASK — sub-question, có thêm findings
# ============================================================

TaskStatus = Literal["pending", "in_progress", "completed", "needs_more", "failed"]


class ResearchTask(BaseModel):
    id: int
    sub_question: str
    description: str
    status: TaskStatus = "pending"
    findings: Optional[str] = None


class SearchQuery(BaseModel):
    query: str
    timestamp: str


class EvidenceModel(BaseModel):
    text: str
    source_url: str
    title: str
    score: float


def merge_evidence(left: List[EvidenceModel], right: List[EvidenceModel]) -> List[EvidenceModel]:
    """Custom reducer for collected_evidence to merge lists of evidence while preserving uniqueness.
    
    Deduplicates by (source_url, hash(text[:500])) to allow similar chunks from different
    sources while preventing true duplicates from the same source.
    """
    if not left:
        left = []
    if not right:
        right = []
    
    seen = set()
    result = []
    
    def _dedup_key(ev: EvidenceModel) -> tuple:
        return (ev.source_url, hash(ev.text[:500]))
    
    for ev in left:
        key = _dedup_key(ev)
        if key not in seen:
            seen.add(key)
            result.append(ev)
            
    for ev in right:
        key = _dedup_key(ev)
        if key not in seen:
            seen.add(key)
            result.append(ev)
            
    return result


class Citation(BaseModel):
    index: int
    url: str
    title: str


class WrittenReport(BaseModel):
    title: str
    report_body: str
    citations: List[Citation]


class AgentState(TypedDict):
    # Input
    user_query: str
    config: Dict[str, Any]
    
    # Planning
    research_context: Optional[ResearchContext]
    research_plan: List[ResearchTask]
    current_task_idx: int
    
    # Research
    search_history: List[SearchQuery]
    collected_evidence: Annotated[List[EvidenceModel], merge_evidence]
    
    # Review
    review_feedback: Optional[str]
    reflection_count: int
    
    # Writing
    draft_sections: Dict[str, str]
    final_report: Optional[str]
    citations: List[Citation]
    
    # Control
    step_count: int
    status: Literal["planning", "researching", "reviewing", "writing", "done", "error"]
    error_log: Annotated[List[str], operator.add]
    reviewer_decisions: Annotated[List[Dict[str, Any]], operator.add]
