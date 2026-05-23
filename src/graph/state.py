from typing import Dict, List, Literal, Optional, TypedDict, Any
from pydantic import BaseModel, Field

class ResearchTask(BaseModel):
    id: int
    sub_question: str
    description: str
    status: Literal["pending", "completed", "failed"] = "pending"
    findings: Optional[str] = None

class SearchQuery(BaseModel):
    query: str
    timestamp: str

class EvidenceModel(BaseModel):
    text: str
    source_url: str
    title: str
    score: float

class Citation(BaseModel):
    index: int
    url: str
    title: str

class WrittenReport(BaseModel):
    title: str
    report_body: str
    references: List[Citation]

class AgentState(TypedDict):
    # Input
    user_query: str
    config: Dict[str, Any]
    
    # Planning
    research_plan: List[ResearchTask]
    current_task_idx: int
    
    # Research
    search_history: List[SearchQuery]
    collected_evidence: List[EvidenceModel]
    
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
    error_log: List[str]
