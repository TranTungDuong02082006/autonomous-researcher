import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

class TraceLogger:
    def __init__(self, trace_dir: str = "logs/traces"):
        self.trace_dir = trace_dir
        os.makedirs(self.trace_dir, exist_ok=True)
        self.steps = []
        self.tool_calls = []
        self.llm_calls = []
        self.start_time = time.time()

    def log_step(self, agent_name: str, action: str, input_data: Any, output_data: Any, metadata: Optional[Dict[str, Any]] = None):
        """Log a high-level agent transition step."""
        step = {
            "timestamp": datetime.now().isoformat(),
            "agent_name": agent_name,
            "action": action,
            "input": input_data,
            "output": output_data,
            "metadata": metadata or {}
        }
        self.steps.append(step)
        logger.info(f"[{agent_name}] {action}")

    def log_tool_call(self, tool_name: str, args: Any, result: Any, duration: float):
        """Log a tool execution and its timing."""
        tool_call = {
            "timestamp": datetime.now().isoformat(),
            "tool_name": tool_name,
            "arguments": args,
            "result": str(result)[:2000] + "..." if len(str(result)) > 2000 else str(result),
            "duration_seconds": duration
        }
        self.tool_calls.append(tool_call)
        logger.info(f"[Tool: {tool_name}] executed in {duration:.2f}s")

    def log_llm_call(self, messages: List[Dict[str, str]], response: str, tokens_used: int = 0):
        """Log a chat generation API transaction."""
        llm_call = {
            "timestamp": datetime.now().isoformat(),
            "prompt_messages": messages,
            "response_content": response,
            "tokens_used_estimate": tokens_used
        }
        self.llm_calls.append(llm_call)
        logger.info(f"[LLM Call] Generated response of {len(response)} chars")

    def export(self, query: str, format_type: str = "json") -> str:
        """Compile and save trace history to file."""
        duration = time.time() - self.start_time
        filename = f"trace_{int(time.time())}.json"
        filepath = os.path.join(self.trace_dir, filename)

        full_trace = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": duration,
            "summary": {
                "total_steps": len(self.steps),
                "total_tool_calls": len(self.tool_calls),
                "total_llm_calls": len(self.llm_calls)
            },
            "steps": self.steps,
            "tool_calls": self.tool_calls,
            "llm_calls": self.llm_calls
        }

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(full_trace, f, ensure_ascii=False, indent=2)
            logger.info(f"Trace exported successfully to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to export trace: {e}")
            return ""
