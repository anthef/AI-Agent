"""
Solution A: Trace container with @observe, but NO Gemini call tracing.

You:
- Already have streamed logs (OpenAI/LangChain style).
- You replay those logs into observed spans so DeepEval builds a real trace.
- PlanQualityMetric runs via evals_iterator (required by DeepEval).

Refs:
- PlanQualityMetric must be used in observe or evals_iterator. (DeepEval docs)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import json
import os

from dotenv import load_dotenv

from deepeval.tracing import observe, update_current_span, update_current_trace, trace_manager
from deepeval.metrics import PlanQualityMetric
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.test_case import ToolCall

from task_2.gemini_evaluator import GeminiEvaluator


load_dotenv()


# ----------------------------
# 1) Event schema (your stream log adapter target)
# ----------------------------
@dataclass
class StreamEvent:
    kind: str  # "planning" | "tool" | "final"
    text: Optional[str] = None

    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_output: Optional[Any] = None


def parse_events_from_openai_like_log(log: List[Dict[str, Any]]) -> List[StreamEvent]:
    """
    Minimal adapter example.
    Input: list of dict events from your streaming collector.
    You can change this to match your real collector output.
    """
    events: List[StreamEvent] = []

    for item in log:
        t = item.get("type") or item.get("event") or item.get("kind")

        if t in ("planning", "plan", "reasoning", "thinking"):
            events.append(StreamEvent(kind="planning", text=item.get("text") or item.get("content") or ""))

        elif t in ("tool_call", "tool", "function_call"):
            events.append(
                StreamEvent(
                    kind="tool",
                    tool_name=item.get("name") or item.get("tool_name"),
                    tool_args=item.get("args") or item.get("arguments") or item.get("input") or {},
                    tool_output=item.get("output") or item.get("result"),
                )
            )

        elif t in ("final", "assistant_final", "completion"):
            events.append(StreamEvent(kind="final", text=item.get("text") or item.get("content") or ""))

    return events


# ----------------------------
# 2) Observed spans that we will "replay" into
# ----------------------------
@observe(name="thinking", type="agent")
def planning_span(planning_text: str) -> str:
    # Force span name to be exactly "thinking" even if function name differs
    update_current_span(name="thinking")
    return planning_text


@observe(type="tool")
def tool_span(tool_name: str, tool_args: Dict[str, Any], tool_output: Any) -> Any:
    # Make each tool span show the actual tool name
    update_current_span(
        name=tool_name,
        metadata={
            "input_parameters": tool_args,
            "output": tool_output,
        },
    )
    return tool_output


@observe(name="agent", type="agent")
def agent_replay_from_events(user_input: str, events: List[StreamEvent]) -> str:
    """
    This function creates the real trace:
    - planning span output contains the plan text
    - tool spans represent tool calls
    - trace-level fields set by update_current_trace()
    """
    plan_text = ""
    final_text = ""

    tools_called: List[ToolCall] = []

    for e in events:
        if e.kind == "planning" and e.text is not None:
            plan_text = e.text.strip()
            if plan_text:
                planning_span(plan_text)

        elif e.kind == "tool":
            name = e.tool_name or "unknown_tool"
            args = e.tool_args or {}
            out = e.tool_output
            tool_span(name, args, out)

            tools_called.append(
                ToolCall(
                    name=name,
                    input_parameters=args,
                    output=out,
                )
            )

        elif e.kind == "final" and e.text is not None:
            final_text = e.text.strip()

    if not final_text:
        final_text = "OK"

    update_current_trace(
        input=user_input,
        output=final_text,
        tools_called=tools_called,
        name="ReplayAgentTrace",
        metadata={
            "source": "replayed_stream_logs",
            "has_plan": bool(plan_text),
        },
    )

    return final_text


# ----------------------------
# 3) Run eval with evals_iterator (required)
# ----------------------------
def main():
    evaluator_model = GeminiEvaluator(model_name=os.getenv("EVAL_MODEL", "gemini-2.5-flash"))

    metric = PlanQualityMetric(
        threshold=0.5,
        model=evaluator_model,
        include_reason=True,
    )

    user_input = "Order 2 laptops to Jakarta"

    # Simulated streaming logs from your agent/LLM framework
    # In production, this would come from your actual streaming collector
    openai_like_log = [
        {"type": "planning", "text": "1. Check inventory for laptop product\n2. Verify at least 2 units\n3. Process payment and arrange shipping to Jakarta"},
        {"type": "tool_call", "name": "check_inventory", "args": {"product": "laptop", "quantity": 2}, "output": {"status": "available", "stock": 15}},
        {"type": "tool_call", "name": "process_payment", "args": {"amount": 2000, "destination": "Jakarta"}, "output": {"success": True, "transaction_id": "TXN123"}},
        {"type": "final", "text": "Order placed successfully. Transaction ID: TXN123"},
    ]
    events = parse_events_from_openai_like_log(openai_like_log)

    trace_manager.clear_traces()

    dataset = EvaluationDataset(
        goldens=[Golden(input=user_input)]
    )

    # Critical: PlanQualityMetric MUST run via evals_iterator or observe. (DeepEval docs)
    for golden in dataset.evals_iterator(metrics=[metric]):
        _ = agent_replay_from_events(golden.input, events)

    # Optional debug: inspect trace structure locally
    traces = trace_manager.traces
    print(f"\n=== DEBUG TRACE INFO ===")
    print(f"Traces captured: {len(traces)}")

    if traces:
        last = traces[-1]
        print("Last trace name:", getattr(last, "name", None))

    print("\n=== HASIL EVALUASI ===")
    print("PlanQualityMetric score:", metric.score)
    print("PlanQualityMetric reason:", metric.reason)


if __name__ == "__main__":
    main()
