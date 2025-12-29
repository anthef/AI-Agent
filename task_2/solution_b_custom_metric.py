"""
Solution B: No @observe at all.

Flow:
- Feed stream logs directly.
- Extract task + plan from logs.
- Use LLM judge (Gemini) to score alignment 0..1.
- This is a custom metric. Not PlanQualityMetric built-in.

Authors:
    Anthony Edbert Feriyanto

References:
- Custom metrics should inherit BaseMetric and implement measure/a_measure. (DeepEval docs)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import asyncio
import json
import os
import re

from dotenv import load_dotenv

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from task_2.gemini_evaluator import GeminiEvaluator


load_dotenv()


def _extract_email(text: str) -> str | None:
    """Extract email address from text."""
    m = re.search(r'([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})', text)
    return m.group(1) if m else None


def _extract_quantity_and_product(text: str) -> tuple[int | None, str | None]:
    """Extract quantity and product from text."""
    t = text.lower()

    qty = None
    m = re.search(r'\b(\d+)\b', t)
    if m:
        try:
            qty = int(m.group(1))
        except Exception:
            qty = None

    product = None
    if "laptop" in t:
        product = "LAPTOP-001"
    elif "mouse" in t:
        product = "MOUSE-002"
    elif "keyboard" in t:
        product = "KEYBOARD-003"
    elif "monitor" in t:
        product = "MONITOR-004"
    elif "headset" in t or "headphone" in t:
        product = "HEADSET-005"

    return qty, product


def _extract_destination(text: str) -> str | None:
    """Extract destination city from text."""
    t = text.lower()

    known = ["jakarta", "bandung", "surabaya", "bali", "singapore", "medan", "semarang"]
    for k in known:
        if k in t:
            return k[0].upper() + k[1:]

    m = re.search(r'\b(to|ke)\s+([A-Z][a-z]+)\b', text)
    if m:
        return m.group(2)

    return None


def _extract_discount_code(text: str) -> str | None:
    """Extract discount code from text."""
    t = text

    m = re.search(r'(discount\s*code|kode\s*diskon)\s*[:\-]?\s*([A-Za-z0-9_-]+)', t, flags=re.IGNORECASE)
    if m:
        return m.group(2)

    m2 = re.search(r'\b([A-Z]{4,}\d{1,})\b', t)
    return m2.group(1) if m2 else None


def _extract_payment_method(text: str) -> str | None:
    """Extract payment method from text."""
    t = text.lower()
    if "credit card" in t or "kartu kredit" in t:
        return "credit_card"
    if "debit" in t:
        return "debit_card"
    if "gopay" in t:
        return "gopay"
    if "ovo" in t:
        return "ovo"
    if "bank transfer" in t or "transfer" in t:
        return "bank_transfer"
    if "ewallet" in t or "e-wallet" in t:
        return "ewallet"
    return None


def synthesize_plan_from_task(task: str, tool_names: list[str]) -> str:
    """Generate a semantic plan from task and tool calls.
    
    Args:
        task (str): The original task/input.
        tool_names (list[str]): List of tools that were called.
        
    Returns:
        str: A detailed, semantic plan.
    """
    seen = set()
    unique_tools = []
    for tool in tool_names:
        if tool not in seen:
            seen.add(tool)
            unique_tools.append(tool)
    
    qty, product = _extract_quantity_and_product(task)
    dest = _extract_destination(task)
    code = _extract_discount_code(task)
    pay = _extract_payment_method(task)
    email = _extract_email(task)

    qty_text = str(qty) if qty is not None else "the requested quantity"
    product_text = product if product is not None else "the requested product"
    dest_text = dest if dest is not None else "the destination city"
    code_text = code if code else "no discount code"
    pay_text = pay if pay else "the requested payment method"
    email_text = email if email else "the customer email"

    steps: list[str] = []

    steps.append(
        f"Parse request: product={product_text}, quantity={qty_text}, destination={dest_text}, "
        f"discount_code={code_text}, payment_method={pay_text}, email={email_text}."
    )

    tool_descriptions = {
        "check_inventory": f"Check inventory availability for {product_text} with quantity {qty_text}.",
        "apply_discount": f"Apply discount code {code_text} to the total price." if code else "Process pricing (no discount code provided).",
        "calculate_shipping": f"Calculate shipping cost to {dest_text} based on weight.",
        "process_payment": f"Process payment using {pay_text} for the final total amount.",
        "send_confirmation_email": f"Send order confirmation email to {email_text}.",
    }

    for tool in unique_tools:
        desc = tool_descriptions.get(tool, f"Execute {tool}.")
        steps.append(desc)

    return "\n".join([f"{i+1}. {s}" for i, s in enumerate(steps)])


@dataclass
class StreamEvent:
    kind: str  
    text: str | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_output: Any | None = None


def extract_plan_text(events: list[StreamEvent], task: str = "") -> str:
    """Extract planning text from event stream.

    Args:
        events (list[StreamEvent]): List of stream events.
        task (str): Original task for semantic fallback.

    Returns:
        str: The extracted plan text, or semantic fallback from tools.
    """
    for e in events:
        if e.kind == "planning" and e.text:
            return e.text.strip()

    tools = [e for e in events if e.kind == "tool" and e.tool_name]
    if not tools:
        return ""

    if task:
        tool_names = [t.tool_name for t in tools if t.tool_name]
        return synthesize_plan_from_task(task, tool_names)

    tool_descriptions = {
        "check_inventory": "Check product inventory and availability",
        "apply_discount": "Apply discount code to reduce total price",
        "calculate_shipping": "Calculate shipping cost to destination",
        "process_payment": "Process payment and get transaction confirmation",
        "send_confirmation_email": "Send order confirmation email to customer",
    }

    lines = []
    for i, t in enumerate(tools, start=1):
        desc = tool_descriptions.get(t.tool_name, f"Execute {t.tool_name}")
        lines.append(f"{i}. {desc}")
    return "\n".join(lines)


def extract_task_text(test_case: LLMTestCase) -> str:
    """Extract task text from test case.

    Args:
        test_case (LLMTestCase): The test case containing input.

    Returns:
        str: The extracted task text.
    """
    return str(getattr(test_case, "input", "") or "")


def extract_tools_summary(events: list[StreamEvent]) -> str:
    """Extract tools summary from event stream.

    Args:
        events (list[StreamEvent]): List of stream events.

    Returns:
        str: JSON string summary of tools called.
    """
    def truncate_output(obj: Any, max_len: int = 500) -> Any:
        """Truncate large outputs to prevent token bloat."""
        if obj is None:
            return obj
        s = str(obj)
        if len(s) > max_len:
            return s[:max_len] + "...(truncated)"
        return obj
    
    tools = []
    for e in events:
        if e.kind == "tool":
            tools.append(
                {
                    "name": e.tool_name,
                    "args": e.tool_args,
                    "output": truncate_output(e.tool_output),
                }
            )
    return json.dumps(tools, ensure_ascii=False)


def safe_json_extract(text: str) -> dict[str, Any]:
    """Try parse JSON from model output.

    Accepts raw JSON or JSON embedded in text.

    Args:
        text (str): The text containing JSON.

    Returns:
        dict[str, Any]: The parsed JSON dict, or empty dict if failure.
    """
    text = text.strip()
    
    try:
        return json.loads(text)
    except Exception:
        pass

    patterns = [
        r"\{.*?\}",  
        r"\{[^}]*\}",  
        r"\{.*\}",  
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.DOTALL)
        for match in matches:
            try:
                result = json.loads(match)
                if "score" in result:
                    return result
            except Exception:
                continue
    
    return {}


class PlanQualityFromLogsMetric(BaseMetric):
    """Custom metric to evaluate plan quality from logs."""

    def __init__(
        self,
        threshold: float = 0.5,
        model_name: str = "gemini-2.5-flash",
        include_reason: bool = True,
    ):
        """Initialize the metric.

        Args:
            threshold (float, optional): Score threshold. Defaults to 0.5.
            model_name (str, optional): Model for evaluation. Defaults to "gemini-2.5-flash".
            include_reason (bool, optional): Whether to include reason. Defaults to True.
        """
        self.threshold = threshold
        self.include_reason = include_reason
        self.evaluation_model = model_name
        self.model = GeminiEvaluator(model_name=model_name)

        self.score = 0.0
        self.reason = ""
        self.success = False
        self.error = None

    @property
    def name(self) -> str:
        """Metric name for DeepEval reporting."""
        return "PlanQualityFromLogs"

    def _get_events(self, test_case: LLMTestCase) -> list[StreamEvent]:
        events = getattr(test_case, "_events", None) or getattr(test_case, "events", None)
        if not events:
            return []
        return events

    def _judge(self, task: str, plan: str, tools_json: str, final_output: str) -> dict[str, Any]:
        """Use LLM to judge plan quality.
        
        Args:
            task (str): The original task.
            plan (str): The plan to evaluate.
            tools_json (str): JSON summary of tools.
            final_output (str): The final output.
            
        Returns:
            dict[str, Any]: Dictionary with 'score' and 'reason'.
        """
        prompt = (
            "You are an evaluator for agentic system plans.\n"
            "Evaluate how well the plan accomplishes the given task.\n\n"
            "Criteria:\n"
            "- Completeness: Does the plan cover all necessary steps?\n"
            "- Correctness: Are the steps in logical order?\n"
            "- Feasibility: Can this plan realistically complete the task?\n"
            "- Specificity: Does the plan reference specific details from the task?\n\n"
            "Return ONLY valid JSON with exactly these keys: {\"score\": <float 0-1>, \"reason\": \"<explanation>\"}.\n"
            "If you output anything other than valid JSON, your response will be discarded and scored 0.\n\n"
            f"Task:\n{task}\n\n"
            f"Plan:\n{plan}\n\n"
            f"Tools Executed:\n{tools_json}\n\n"
            f"Final Output:\n{final_output}\n\n"
            "JSON output:"
        )

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
        """Measure the metric for a test case.

        Args:
            test_case (LLMTestCase): The test case to evaluate.

        Returns:
            float: The calculated score.

        Raises:
            Exception: If measurement fails.
        """
        try:
            events = self._get_events(test_case)
            task = extract_task_text(test_case)
            
            plan = extract_plan_text(events, task)
            tools_json = extract_tools_summary(events)
            
            final_output = str(getattr(test_case, "actual_output", "") or "")
            if not final_output:
                for e in events:
                    if e.kind == "final" and e.text:
                        final_output = e.text.strip()
                        break

            if not plan.strip():
                self.score = 0.0
                self.reason = "No plan found in logs and no tool calls to generate fallback plan."
                self.success = False
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
        """Measure command asynchronously.

        Args:
            test_case (LLMTestCase): The test case.

        Returns:
            float: The score.
        """
        return await asyncio.to_thread(self.measure, test_case)


def main():
    """Main execution entry point."""
    user_input = "I want to order 2 laptops to Jakarta, use discount code WELCOME10, pay with credit card, email: john@example.com"
    final_output = "Order placed successfully. Transaction ID: TXN456"

    events = [
        StreamEvent(
            kind="planning",
            text=(
                "1. Parse request: product=LAPTOP-001, quantity=2, destination=Jakarta, discount_code=WELCOME10, payment_method=credit_card, email=john@example.com.\n"
                "2. Check inventory availability for LAPTOP-001 with quantity 2.\n"
                "3. Apply discount code WELCOME10 to the total price.\n"
                "4. Calculate shipping cost to Jakarta based on weight.\n"
                "5. Process payment using credit_card for the final total amount.\n"
                "6. Send order confirmation email to john@example.com."
            )
        ),
        StreamEvent(kind="tool", tool_name="check_inventory", tool_args={"product_id": "LAPTOP-001", "quantity": 2}, tool_output={"available": True, "stock": 50, "price": 1200, "total_price": 2400}),
        StreamEvent(kind="tool", tool_name="apply_discount", tool_args={"total_price": 2400, "discount_code": "WELCOME10"}, tool_output={"original_price": 2400, "discount_amount": 240, "final_price": 2160}),
        StreamEvent(kind="tool", tool_name="calculate_shipping", tool_args={"destination_city": "Jakarta", "total_weight_kg": 5.0}, tool_output={"available": True, "cost": 50, "destination": "Jakarta"}),
        StreamEvent(kind="tool", tool_name="process_payment", tool_args={"amount": 2210, "payment_method": "credit_card"}, tool_output={"success": True, "transaction_id": "TXN456"}),
        StreamEvent(kind="tool", tool_name="send_confirmation_email", tool_args={"customer_email": "john@example.com", "order_summary": {"product": "LAPTOP-001", "quantity": 2, "total": 2210}}, tool_output={"sent": True, "email": "john@example.com"}),
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

    print("\n=== TEST 1: With explicit detailed plan ===")
    print("Score:", metric.score)
    print("Reason:", metric.reason)
    print("Success:", metric.success)

    print("\n=== TEST 2: No explicit plan - semantic fallback from tools ===")
    
    events2 = [
        StreamEvent(kind="tool", tool_name="check_inventory", tool_args={"product_id": "LAPTOP-001", "quantity": 2}, tool_output={"available": True, "stock": 50}),
        StreamEvent(kind="tool", tool_name="apply_discount", tool_args={"total_price": 2400, "discount_code": "WELCOME10"}, tool_output={"final_price": 2160}),
        StreamEvent(kind="tool", tool_name="calculate_shipping", tool_args={"destination_city": "Jakarta", "total_weight_kg": 5.0}, tool_output={"cost": 50}),
        StreamEvent(kind="tool", tool_name="process_payment", tool_args={"amount": 2210, "payment_method": "credit_card"}, tool_output={"success": True, "transaction_id": "TXN789"}),
        StreamEvent(kind="tool", tool_name="send_confirmation_email", tool_args={"customer_email": "john@example.com"}, tool_output={"sent": True}),
    ]
    
    test_case2 = LLMTestCase(
        input=user_input,
        actual_output="Order completed successfully",
    )
    test_case2._events = events2
    
    metric2 = PlanQualityFromLogsMetric(
        threshold=0.5,
        model_name=os.getenv("EVAL_MODEL", "gemini-2.5-flash"),
        include_reason=True,
    )
    
    plan_fallback = extract_plan_text(events2, user_input)
    print("Generated Plan (fallback):")
    print(plan_fallback)
    
    metric2.measure(test_case2)
    print("\nScore:", metric2.score)
    print("Reason:", metric2.reason)
    print("Success:", metric2.success)
    
    print("\n=== TEST 3: Empty events - should fail with score 0 ===")
    
    test_case3 = LLMTestCase(
        input="Order something",
        actual_output="Done",
    )
    test_case3._events = []
    
    metric3 = PlanQualityFromLogsMetric(
        threshold=0.5,
        model_name=os.getenv("EVAL_MODEL", "gemini-2.5-flash"),
        include_reason=True,
    )
    
    metric3.measure(test_case3)
    print("Score:", metric3.score)
    print("Reason:", metric3.reason)
    print("Success:", metric3.success)


if __name__ == "__main__":
    main()
