import logging
import time
from typing import List
from pydantic import BaseModel, Field
from src.models.llm_server import LLMClient
from src.graph.state import ResearchTask
from src.utils.tracer import TraceLogger

logger = logging.getLogger(__name__)

class ResearchPlanSchema(BaseModel):
    plan: List[ResearchTask] = Field(description="A list of 3-7 sequenced sub-questions/tasks to complete the research.")

class Planner:
    def __init__(self, llm_client: LLMClient, tracer: TraceLogger):
        self.llm_client = llm_client
        self.tracer = tracer

    def plan(self, query: str) -> List[ResearchTask]:
        """Formulate a step-by-step sequential research plan."""
        logger.info(f"Generating research plan for query: '{query}'")
        
        system_prompt = (
            "You are a top-tier Principal Research Planner. Your job is to receive a complex, macro research "
            "question and break it down into a logical sequence of 3 to 7 smaller, concrete sub-questions/tasks (System-2 Planning). "
            "Each sub-question must target a specific facet of the user's problem. "
            "Assign unique ascending integers as IDs (1, 2, 3...) to tasks in the sequence."
        )

        user_prompt = (
            f"Divide the following complex query into 3 to 7 specific, actionable sub-tasks/questions:\n"
            f"'{query}'"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        t0 = time.time()
        try:
            # Enforce structured Pydantic schema return
            structured_data = self.llm_client.generate_structured(messages, schema=ResearchPlanSchema)
            
            # Reconstruct list of ResearchTask instances
            tasks = []
            for task_dict in structured_data.get("plan", []):
                # Ensure pending state by default
                task_dict["status"] = "pending"
                tasks.append(ResearchTask(**task_dict))

            duration = time.time() - t0
            self.tracer.log_tool_call("Planner.plan", {"query": query}, f"Generated plan with {len(tasks)} tasks", duration)
            self.tracer.log_step("Planner", "Generated research plan", query, [t.model_dump() for t in tasks])
            
            return tasks
        except Exception as e:
            logger.error(f"Failed to generate structured plan: {e}. Falling back to a simple single-step plan.")
            # Fallback to single task plan
            duration = time.time() - t0
            fallback_task = ResearchTask(
                id=1,
                sub_question=query,
                description="Investigate the user query directly.",
                status="pending"
            )
            self.tracer.log_tool_call("Planner.plan_fallback", {"query": query}, "Generated fallback task", duration)
            return [fallback_task]
