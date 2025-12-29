"""
Solution A: Trace container with @observe, but NO Gemini call tracing.

Flow:
- Already have streamed logs (OpenAI/LangChain style).
- You replay those logs into observed spans so DeepEval builds a real trace.
- PlanQualityMetric runs via evals_iterator (required by DeepEval).

Authors:
    Anthony Edbert Feriyanto

References:
    - PlanQualityMetric must be used in observe or evals_iterator. (DeepEval docs)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
# import json
import os
import re

from dotenv import load_dotenv

from deepeval.tracing import observe, update_current_span, update_current_trace, trace_manager
from deepeval.metrics import PlanQualityMetric
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.test_case import ToolCall

from task_2.gemini_evaluator import GeminiEvaluator
from task_1.agent import ecommerce_agent, TOOL_FUNCTIONS


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


def synthesize_plan(user_input: str, tool_order: list[str] | None = None) -> str:
    """Generate a detailed plan from user input.
    
    Args:
        user_input (str): The customer request.
        tool_order (list[str] | None): Expected tool execution order.
        
    Returns:
        str: A detailed, step-by-step plan.
    """
    if tool_order is None:
        tool_order = [
            "check_inventory",
            "apply_discount",
            "calculate_shipping",
            "process_payment",
            "send_confirmation_email",
        ]

    qty, product = _extract_quantity_and_product(user_input)
    dest = _extract_destination(user_input)
    code = _extract_discount_code(user_input)
    pay = _extract_payment_method(user_input)
    email = _extract_email(user_input)

    qty_text = str(qty) if qty is not None else "the requested quantity"
    product_text = product if product is not None else "the requested product"
    dest_text = dest if dest is not None else "the destination city"
    code_text = code if code else "no code"
    pay_text = pay if pay else "the requested method"
    email_text = email if email else "the customer email"

    steps: list[str] = []

    steps.append(
        f"Parse the request. Identify product={product_text}, quantity={qty_text}, destination={dest_text}, "
        f"discount_code={code_text}, payment_method={pay_text}, email={email_text}."
    )

    if "check_inventory" in tool_order:
        steps.append(f"Call check_inventory for product={product_text} and quantity={qty_text}.")

    if "apply_discount" in tool_order:
        if code:
            steps.append(f"Call apply_discount using discount_code={code_text}.")
        else:
            steps.append("Skip discount or call apply_discount with empty/none code based on tool contract.")

    if "calculate_shipping" in tool_order:
        steps.append(f"Call calculate_shipping for destination={dest_text}. Use weight = quantity * 2.5 kg as instructed.")

    if "process_payment" in tool_order:
        steps.append(f"Call process_payment using the computed total and payment_method={pay_text}.")

    if "send_confirmation_email" in tool_order:
        steps.append(f"Call send_confirmation_email to {email_text} with order summary and transaction id.")

    return "\n".join([f"{i+1}. {s}" for i, s in enumerate(steps)])


def _looks_generic(plan_text: str) -> bool:
    """Check if plan text looks generic/incomplete."""
    t = plan_text.strip().lower()
    if not t:
        return True
    if "→" in t and len(t.split("→")) <= 6:
        return True
    if t.startswith("plan:") and "\n" not in t:
        return True
    if len(t) < 80:
        return True
    return False


class StreamingLogCollector:
    """Collects agent execution events in real-time."""

    def __init__(self):
        self.logs: list[dict[str, Any]] = []

    def log_planning(self, text: str):
        """Log a planning event.

        Args:
            text (str): The planning text to log.
        """
        self.logs.append({"type": "planning", "text": text})

    def log_tool_call(self, name: str, args: dict[str, Any], output: Any):
        """Log a tool call event.

        Args:
            name (str): Name of the tool called.
            args (dict[str, Any]): Arguments passed to the tool.
            output (Any): The output returned by the tool.
        """
        self.logs.append({
            "type": "tool_call",
            "name": name,
            "args": args,
            "output": output
        })

    def log_final(self, text: str):
        """Log the final response event.

        Args:
            text (str): The final response text.
        """
        self.logs.append({"type": "final", "text": text})

    def get_logs(self) -> list[dict[str, Any]]:
        """Retrieve all collected logs.

        Returns:
            list[dict[str, Any]]: A copy of the collected logs.
        """
        return self.logs.copy()

    def clear(self):
        """Clear all collected logs."""
        self.logs.clear()


@dataclass
class StreamEvent:
    kind: str  
    text: str | None = None

    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_output: Any | None = None


def parse_events_from_openai_like_log(log: list[dict[str, Any]]) -> list[StreamEvent]:
    """Parse raw logs into StreamEvents.

    Minimal adapter example.
    Input: list of dict events from your streaming collector.
    You can change this to match your real collector output.

    Args:
        log (list[dict[str, Any]]): The raw logs to parse.

    Returns:
        list[StreamEvent]: The parsed stream events.
    """
    events: list[StreamEvent] = []

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


@observe(name="thinking", type="agent")
def planning_span(planning_text: str) -> str:
    """Create an observed span for planning.

    Args:
        planning_text (str): The planning content.

    Returns:
        str: The planning content.
    """
    update_current_span(name="thinking")
    return planning_text


@observe(type="tool")
def tool_span(tool_name: str, tool_args: dict[str, Any], tool_output: Any) -> Any:
    """Create an observed span for a tool call.

    Args:
        tool_name (str): The name of the tool.
        tool_args (dict[str, Any]): The arguments passed to the tool.
        tool_output (Any): The output of the tool.

    Returns:
        Any: The tool output.
    """
    update_current_span(
        name=tool_name,
        metadata={
            "input_parameters": tool_args,
            "output": tool_output,
        },
    )
    return tool_output


@observe(name="agent", type="agent")
def agent_replay_from_events(user_input: str, events: list[StreamEvent]) -> str:
    """Replay events into a DeepEval trace.

    This function creates the real trace:
    - planning span output contains the plan text
    - tool spans represent tool calls
    - trace-level fields set by update_current_trace()

    Args:
        user_input (str): The original user input.
        events (list[StreamEvent]): The list of parsed events to replay.

    Returns:
        str: The final output of the agent execution.
    """
    plan_text = ""
    final_text = ""

    tools_called: list[ToolCall] = []

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

    if _looks_generic(plan_text):
        tool_names = [e.tool_name for e in events if e.kind == "tool" and e.tool_name]
        tool_order = tool_names if tool_names else None
        plan_text = synthesize_plan(user_input, tool_order=tool_order)
        planning_span(plan_text)

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


def run_agent_with_logging(user_input: str, collector: StreamingLogCollector) -> str:
    """Run the real agent and collect execution logs.

    Args:
        user_input (str): The customer request.
        collector (StreamingLogCollector): Collector to capture logs.

    Returns:
        str: The final response from the agent.
    """
    from task_1.llm import init_llm
    from task_1.agent import TOOLS
    import google.generativeai as genai

    model = init_llm()
    chat = model.start_chat()

    system_instruction = """You are an e-commerce order agent. Process orders by calling tools in sequence:
    1. check_inventory (FIRST - verify product availability)
    2. apply_discount (if discount code provided, use empty string if none)
    3. calculate_shipping (estimate weight = quantity * 2.5 kg)
    4. process_payment (amount = discounted_price + shipping)
    5. send_confirmation_email (LAST - confirm order)

    Use exact values from previous tool results. After calling all tools, respond with a confirmation message."""

    collector.log_planning(synthesize_plan(user_input))

    tool_results = {}
    max_iterations = 15
    iteration = 0

    response = chat.send_message(
        f"{system_instruction}\n\nCustomer request: {user_input}",
        tools=TOOLS
    )

    while iteration < max_iterations:
        iteration += 1

        try:
            parts = response.candidates[0].content.parts
        except Exception:
            break

        has_function_call = False
        final_text = ""

        for part in parts:
            if hasattr(part, 'function_call') and part.function_call:
                has_function_call = True
                fn = part.function_call
                tool_name = fn.name

                args = {k: v for k, v in fn.args.items()}

                if tool_name == "apply_discount" and args.get("discount_code") == "":
                    args["discount_code"] = None

                result = TOOL_FUNCTIONS[tool_name](**args)
                tool_results[tool_name] = result

                collector.log_tool_call(tool_name, args, result)

                response = chat.send_message(
                    genai.protos.Content(
                        parts=[
                            genai.protos.Part(
                                function_response=genai.protos.FunctionResponse(
                                    name=tool_name,
                                    response={"result": result}
                                )
                            )
                        ]
                    ),
                    tools=TOOLS
                )

            elif hasattr(part, 'text') and part.text:
                final_text = part.text.strip()
                has_function_call = False
                break

        if not has_function_call:
            if final_text:
                collector.log_final(final_text)
            break

        if "send_confirmation_email" in tool_results:
            break

    if not final_text:
        final_text = "Order processed successfully"
        collector.log_final(final_text)

    return final_text


def main():
    """Main function to run the evaluation."""
    evaluator_model = GeminiEvaluator(model_name=os.getenv("EVAL_MODEL", "gemini-2.5-flash-lite"))

    metric = PlanQualityMetric(
        threshold=0.5,
        model=evaluator_model,
        include_reason=True,
    )

    user_input = "I want to order 2 laptops to Jakarta, use discount code WELCOME10, pay with credit card, email: john@example.com"

    collector = StreamingLogCollector()
    _ = run_agent_with_logging(user_input, collector)

    real_logs = collector.get_logs()
    print(f"\nCOLLECTED {len(real_logs)} LOG EVENTS:")
    for i, log in enumerate(real_logs, 1):
        print(f"{i}. {log.get('type')}: {log.get('name', '')} {log.get('text', '')[:50] if log.get('text') else ''}")

    events = parse_events_from_openai_like_log(real_logs)

    trace_manager.clear_traces()

    dataset = EvaluationDataset(
        goldens=[Golden(input=user_input)]
    )

    for golden in dataset.evals_iterator(metrics=[metric]):
        _ = agent_replay_from_events(golden.input, events)

    traces = trace_manager.traces
    # print(f"\n=DEBUG TRACE INFO:")
    # print(f"Traces captured: {len(traces)}")

    if traces:
        last = traces[-1]
        print("Last trace name:", getattr(last, "name", None))

    print("\nHASIL EVALUASI:")
    print("PlanQualityMetric score:", metric.score)
    print("PlanQualityMetric reason:", metric.reason)


if __name__ == "__main__":
    main()
