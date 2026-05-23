import logging
from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END

from src.graph.state import AgentState, ResearchTask, EvidenceModel
from src.agents.planner import Planner
from src.agents.researcher import Researcher
from src.agents.reviewer import Reviewer
from src.agents.writer import Writer
from src.graph.guardrails import (
    check_max_steps, 
    check_repeated_queries, 
    check_evidence_sufficiency, 
    force_terminate
)

logger = logging.getLogger(__name__)

def build_research_graph(
    planner: Planner,
    researcher: Researcher,
    reviewer: Reviewer,
    writer: Writer,
    max_reflection_loops: int = 3
):
    """
    Compile the StateGraph for the Autonomous AI Researcher system.
    Connects planner, researcher, reviewer, and writer nodes.
    """
    workflow = StateGraph(AgentState)

    # Define Node 1: Planner
    def planner_node(state: AgentState) -> Dict[str, Any]:
        logger.info("--- PLANNER NODE START ---")
        state["status"] = "planning"
        query = state["user_query"]
        plan = planner.plan(query)
        
        return {
            "research_plan": plan,
            "current_task_idx": 0,
            "reflection_count": 0,
            "status": "researching",
            "step_count": state.get("step_count", 0) + 1
        }

    # Define Node 2: Researcher
    def researcher_node(state: AgentState) -> Dict[str, Any]:
        logger.info("--- RESEARCHER NODE START ---")
        step_count = state.get("step_count", 0) + 1
        
        # Enforce check on steps guardrail
        state["step_count"] = step_count
        if check_max_steps(state):
            force_state = force_terminate(state, "Max step count reached.")
            return force_state

        idx = state["current_task_idx"]
        plan = state["research_plan"]

        if idx >= len(plan):
            logger.info("All planned research tasks completed. Transitioning to writing.")
            return {"status": "writing", "step_count": step_count}

        current_task = plan[idx]
        
        # If there's review feedback from a previous reflection loop, append it to the question
        task_query = current_task.sub_question
        feedback = state.get("review_feedback")
        if feedback and state.get("reflection_count", 0) > 0:
            logger.info(f"Researcher utilizing reviewer feedback: '{feedback}'")
            # Create a localized task representation incorporating the feedback
            task_copy = ResearchTask(
                id=current_task.id,
                sub_question=f"{current_task.sub_question} (Focus on: {feedback})",
                description=current_task.description,
                status=current_task.status
            )
            new_evidence = researcher.execute_task(task_copy, state)
        else:
            new_evidence = researcher.execute_task(current_task, state)

        # Merge evidence preserving uniqueness
        existing_evidence = state.get("collected_evidence", [])
        existing_texts = {ev.text for ev in existing_evidence}
        merged_evidence = list(existing_evidence)
        
        for ev in new_evidence:
            if ev.text not in existing_texts:
                merged_evidence.append(ev)

        # Enforce check on repeated query loop guardrail
        if check_repeated_queries(state):
            force_state = force_terminate(state, "Infinite query loop detected.")
            force_state["collected_evidence"] = merged_evidence
            return force_state

        return {
            "collected_evidence": merged_evidence,
            "status": "reviewing",
            "step_count": step_count,
            "review_feedback": None # Reset feedback
        }

    # Define Node 3: Reviewer
    def reviewer_node(state: AgentState) -> Dict[str, Any]:
        logger.info("--- PEER REVIEWER NODE START ---")
        idx = state["current_task_idx"]
        plan = list(state["research_plan"])
        current_task = plan[idx]

        # Gather evidence relevant to this task for critique
        evidence = state.get("collected_evidence", [])
        
        review_result = reviewer.review(evidence, current_task)
        
        reflection_count = state.get("reflection_count", 0)

        if review_result.sufficient:
            logger.info(f"Evidence for task #{current_task.id} is SUFFICIENT. Saving findings.")
            plan[idx].findings = review_result.findings
            plan[idx].status = "completed"
            
            return {
                "research_plan": plan,
                "current_task_idx": idx + 1,
                "reflection_count": 0,
                "review_feedback": None,
                "status": "researching"
            }
        else:
            logger.info(f"Evidence for task #{current_task.id} is INSUFFICIENT.")
            reflection_count += 1
            
            if reflection_count >= max_reflection_loops:
                logger.warning(f"Exceeded max reflection loops ({max_reflection_loops}) for task #{current_task.id}. Forcing compilation of findings.")
                plan[idx].findings = review_result.findings if review_result.findings else "Incomplete findings due to search limits."
                plan[idx].status = "failed"
                
                return {
                    "research_plan": plan,
                    "current_task_idx": idx + 1,
                    "reflection_count": 0,
                    "review_feedback": None,
                    "status": "researching"
                }
            else:
                logger.info(f"Initiating reflection loop {reflection_count}/{max_reflection_loops} with feedback.")
                return {
                    "reflection_count": reflection_count,
                    "review_feedback": review_result.missing_info,
                    "status": "researching"
                }

    # Define Node 4: Writer
    def writer_node(state: AgentState) -> Dict[str, Any]:
        logger.info("--- WRITER NODE START ---")
        report = writer.write(state)
        
        return {
            "final_report": report.report_body,
            "citations": report.citations,
            "status": "done"
        }

    # Add nodes to graph
    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("writer", writer_node)

    # Establish edge flows
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "researcher")

    # Routing from Researcher
    def route_after_research(state: AgentState) -> Literal["reviewer", "writer"]:
        if state["status"] == "writing":
            return "writer"
        if check_evidence_sufficiency(state):
            return "writer"
        return "reviewer"

    workflow.add_conditional_edges(
        "researcher",
        route_after_research
    )

    # Routing from Reviewer
    def route_after_review(state: AgentState) -> Literal["researcher", "writer"]:
        idx = state["current_task_idx"]
        plan = state["research_plan"]
        
        # If all tasks are completed, or we were routed to writer
        if idx >= len(plan) or state["status"] == "writing":
            return "writer"
        return "researcher"

    workflow.add_conditional_edges(
        "reviewer",
        route_after_review
    )

    workflow.add_edge("writer", END)

    # Compile StateGraph
    return workflow.compile()
