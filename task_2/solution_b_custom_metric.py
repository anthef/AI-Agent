"""
Solution B: No @observe at all.

You:
- Feed stream logs directly.
- Extract task + plan from logs.
- Use LLM judge (Gemini) to score alignment 0..1.
- This is a custom metric. Not PlanQualityMetric built-in.

Refs:
- Custom metrics should inherit BaseMetric and implement measure/a_measure. (DeepEval docs)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import asyncio
import json
import os
import re

from dotenv import load_dotenv

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from task_2.gemini_evaluator import GeminiEvaluator


load_dotenv()


@dataclass
class StreamEvent:
    kind: str  # "planning" | "tool" | "final"
    text: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_output: Optional[Any] = None


def extract_plan_text(events: List[StreamEvent]) -> str:
    for e in events:
        if e.kind == "planning" and e.text:
            return e.text.strip()

    # Fallback: build a plan skeleton from tools
    tools = [e for e in events if e.kind == "tool" and e.tool_name]
    if not tools:
        return ""

    lines = []
    for i, t in enumerate(tools, start=1):
        args = t.tool_args or {}
        lines.append(f"{i}. Call tool {t.tool_name} with args {json.dumps(args, ensure_ascii=False)}")
    return "\n".join(lines)


def extract_task_text(test_case: LLMTestCase) -> str:
    # Task biasanya = input user
    return str(getattr(test_case, "input", "") or "")


def extract_tools_summary(events: List[StreamEvent]) -> str:
    tools = []
    for e in events:
        if e.kind == "tool":
            tools.append(
                {
                    "name": e.tool_name,
                    "args": e.tool_args,
                    "output": e.tool_output,
                }
            )
    return json.dumps(tools, ensure_ascii=False)


def safe_json_extract(text: str) -> Dict[str, Any]:
    """
    Try parse JSON from model output.
    Accepts raw JSON or JSON embedded in text.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


class PlanQualityFromLogsMetric(BaseMetric):
    def __init__(
        self,
        threshold: float = 0.5,
        model_name: str = "gemini-2.5-flash",
        include_reason: bool = True,
    ):
        self.threshold = threshold
        self.include_reason = include_reason
        self.evaluation_model = model_name
        self.model = GeminiEvaluator(model_name=model_name)

        self.score = 0.0
        self.reason = ""
        self.success = False
        self.error = None

    def _get_events(self, test_case: LLMTestCase) -> List[StreamEvent]:
        # You can attach events however you want.
        # Supported patterns:
        # - test_case._events
        # - test_case.events
        events = getattr(test_case, "_events", None) or getattr(test_case, "events", None)
        if not events:
            return []
        return events

    def _judge(self, task: str, plan: str, tools_json: str, final_output: str) -> Dict[str, Any]:
        prompt = (
            "You are an evaluator.\n"
            "Give a plan quality score in [0,1] for how well the plan can complete the task.\n"
            "Consider completeness, correct ordering, missing steps, and feasibility.\n"
            "Return JSON only with keys: score, reason.\n\n"
            f"Task:\n{task}\n\n"
            f"Plan:\n{plan}\n\n"
            f"Tools:\n{tools_json}\n\n"
            f"Final output:\n{final_output}\n"
        )

        # GeminiEvaluator must expose a generate method that returns text.
        # If your GeminiEvaluator uses a different method name, change here.
        out_text = self.model.generate(prompt)
        data = safe_json_extract(out_text)

        score = data.get("score", 0.0)
        reason = data.get("reason", "")

        try:
            score = float(score)
        except Exception:
            score = 0.0

        score = max(0.0, min(1.0, score))
        return {"score": score, "reason": str(reason)}

    def measure(self, test_case: LLMTestCase) -> float:
        try:
            events = self._get_events(test_case)
            task = extract_task_text(test_case)
            plan = extract_plan_text(events)
            tools_json = extract_tools_summary(events)
            final_output = str(getattr(test_case, "actual_output", "") or "")

            if not plan:
                # match PlanQuality behavior: pass by default if no plan
                self.score = 1.0
                self.reason = "No explicit plan found in logs. Default pass."
                self.success = True
                return self.score

            judged = self._judge(task, plan, tools_json, final_output)
            self.score = judged["score"]
            self.reason = judged["reason"] if self.include_reason else ""
            self.success = self.score >= self.threshold
            return self.score

        except Exception as e:
            self.error = str(e)
            raise

    async def a_measure(self, test_case: LLMTestCase) -> float:
        # If your GeminiEvaluator supports async, replace with true async call.
        return await asyncio.to_thread(self.measure, test_case)


def main():
    user_input = "Order 2 laptops to Jakarta"
    final_output = "Order placed successfully. Transaction ID: TXN123"

    events = [
        StreamEvent(kind="planning", text="1. Check inventory\n2. Verify 2 units\n3. Process payment and ship to Jakarta"),
        StreamEvent(kind="tool", tool_name="check_inventory", tool_args={"product": "laptop", "quantity": 2}, tool_output={"status": "available", "stock": 15}),
        StreamEvent(kind="tool", tool_name="process_payment", tool_args={"amount": 2000, "destination": "Jakarta"}, tool_output={"success": True, "transaction_id": "TXN123"}),
        StreamEvent(kind="final", text=final_output),
    ]

    test_case = LLMTestCase(
        input=user_input,
        actual_output=final_output,
    )
    test_case._events = events

    metric = PlanQualityFromLogsMetric(
        threshold=0.5,
        model_name=os.getenv("EVAL_MODEL", "gemini-2.5-flash"),
        include_reason=True,
    )

    metric.measure(test_case)

    print("\n=== HASIL EVALUASI ===")
    print("Custom PlanQuality score:", metric.score)
    print("Custom PlanQuality reason:", metric.reason)
    print("Custom PlanQuality success:", metric.success)


if __name__ == "__main__":
    main()
