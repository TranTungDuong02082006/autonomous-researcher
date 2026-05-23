import logging
from typing import Dict, Any
from src.graph.state import AgentState

logger = logging.getLogger(__name__)

def check_max_steps(state: AgentState) -> bool:
    """Check if the agent has reached or exceeded maximum allowable steps."""
    config = state.get("config", {})
    max_steps = config.get("agent", {}).get("max_steps", 15)
    step_count = state.get("step_count", 0)
    
    if step_count >= max_steps:
        logger.warning(f"Guardrail triggered: step count {step_count} >= max steps {max_steps}.")
        return True
    return False

def check_repeated_queries(state: AgentState) -> bool:
    """
    Check if the latest search queries are highly similar to past queries,
    which indicates the agent is stuck in an infinite query loop.
    Uses token Jaccard similarity.
    """
    history = state.get("search_history", [])
    if len(history) < 2:
        return False

    latest_query = history[-1].query.lower().strip()
    latest_words = set(latest_query.split())

    for prev in history[:-1]:
        prev_words = set(prev.query.lower().strip().split())
        if not latest_words or not prev_words:
            continue
        
        # Calculate Jaccard similarity
        intersection = latest_words.intersection(prev_words)
        union = latest_words.union(prev_words)
        jaccard = len(intersection) / len(union)

        if jaccard > 0.85:
            logger.warning(f"Guardrail triggered: Query '{latest_query}' is highly similar to previous query '{prev.query}' (Jaccard: {jaccard:.2f})")
            return True
            
    return False

def check_evidence_sufficiency(state: AgentState) -> bool:
    """Check if we have enough collected evidence overall to skip further research cycles."""
    evidence = state.get("collected_evidence", [])
    # If we have gathered more than 15 evidence chunks, we have plenty to write a strong report
    if len(evidence) >= 15:
        logger.info(f"Guardrail triggered: sufficiency check satisfied with {len(evidence)} evidence chunks.")
        return True
    return False

def force_terminate(state: AgentState, reason: str) -> AgentState:
    """Gracefully terminate agent graph execution due to a guardrail trigger."""
    logger.warning(f"Force terminating agent loop. Reason: {reason}")
    state["status"] = "writing"  # Redirect to writer node to attempt drawing conclusions from what was found
    state["review_feedback"] = f"FORCE TERMINATION: {reason}"
    state["error_log"].append(f"Guardrail forced termination: {reason}")
    return state
