"""Fully Agentic E-Commerce Order Processing Agent.

This module implements a fully agentic system where the LLM decides which tools
to call and when using Gemini's function calling capability.

Authors:
    Anthony Edbert Feriyanto

References:
    NONE
"""

from deepeval.tracing import observe
from dotenv import load_dotenv
import json
import google.generativeai as genai

from task_1.llm import init_llm
from task_1.tools import (
    apply_discount,
    calculate_shipping,
    check_inventory,
    process_payment,
    send_confirmation_email,
)

load_dotenv()

TOOLS = [
    genai.protos.Tool(
        function_declarations=[
            genai.protos.FunctionDeclaration(
                name="check_inventory",
                description="Check product availability and pricing. MUST be called first.",
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        "product_id": genai.protos.Schema(
                            type=genai.protos.Type.STRING,
                            description="Product ID: LAPTOP-001, MOUSE-002, KEYBOARD-003, MONITOR-004, or HEADSET-005",
                        ),
                        "quantity": genai.protos.Schema(
                            type=genai.protos.Type.INTEGER, description="Quantity to order"
                        ),
                    },
                    required=["product_id", "quantity"],
                ),
            ),
            genai.protos.FunctionDeclaration(
                name="apply_discount",
                description="Apply discount code. Call after inventory check if discount code provided.",
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        "total_price": genai.protos.Schema(
                            type=genai.protos.Type.NUMBER, description="Total price before discount"
                        ),
                        "discount_code": genai.protos.Schema(
                            type=genai.protos.Type.STRING,
                            description="Discount code: WELCOME10, SAVE50, or VIP20. Use empty string if none.",
                        ),
                    },
                    required=["total_price"],
                ),
            ),
            genai.protos.FunctionDeclaration(
                name="calculate_shipping",
                description="Calculate shipping cost. Call after discount. Estimate weight as quantity * 2.5 kg.",
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        "destination_city": genai.protos.Schema(
                            type=genai.protos.Type.STRING,
                            description="Destination: Jakarta, Bandung, Surabaya, Bali, or Singapore",
                        ),
                        "total_weight_kg": genai.protos.Schema(
                            type=genai.protos.Type.NUMBER, description="Total weight in kg"
                        ),
                    },
                    required=["destination_city", "total_weight_kg"],
                ),
            ),
            genai.protos.FunctionDeclaration(
                name="process_payment",
                description="Process payment. Call after shipping calculation. Amount = discounted_price + shipping.",
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        "amount": genai.protos.Schema(
                            type=genai.protos.Type.NUMBER, description="Total amount to charge"
                        ),
                        "payment_method": genai.protos.Schema(
                            type=genai.protos.Type.STRING,
                            description="Payment method: credit_card, bank_transfer, or ewallet",
                        ),
                    },
                    required=["amount", "payment_method"],
                ),
            ),
            genai.protos.FunctionDeclaration(
                name="send_confirmation_email",
                description="Send confirmation email. MUST be called last after payment.",
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        "customer_email": genai.protos.Schema(
                            type=genai.protos.Type.STRING, description="Customer email address"
                        ),
                        "order_summary": genai.protos.Schema(
                            type=genai.protos.Type.OBJECT, description="Order summary with all details"
                        ),
                    },
                    required=["customer_email", "order_summary"],
                ),
            ),
        ]
    )
]

TOOL_FUNCTIONS = {
    "check_inventory": check_inventory,
    "apply_discount": apply_discount,
    "calculate_shipping": calculate_shipping,
    "process_payment": process_payment,
    "send_confirmation_email": send_confirmation_email,
}


@observe(type="agent")
def ecommerce_agent(user_input: str, max_iterations: int = 15) -> dict:
    """Fully agentic e-commerce order processing.

    LLM decides which tools to call and when using function calling.

    Args:
        user_input (str): Customer's order request.
        max_iterations (int, optional): Max iterations. Defaults to 15.

    Returns:
        dict: Order processing result.
    """
    model = init_llm()

    chat = model.start_chat()

    system_instruction = """You are an e-commerce order agent. Process orders by calling tools in sequence:
    1. check_inventory (FIRST - verify product availability)
    2. apply_discount (if discount code provided, use empty string if none)
    3. calculate_shipping (estimate weight = quantity * 2.5 kg)
    4. process_payment (amount = discounted_price + shipping)
    5. send_confirmation_email (LAST - confirm order)

    Use exact values from previous tool results. After calling all tools, respond with a confirmation message."""

    print("=" * 80)

    tool_results = {}
    iteration = 0

    try:
        response = chat.send_message(
            f"{system_instruction}\n\nCustomer request: {user_input}", 
            tools=TOOLS
        )
    except Exception as e:
        print(f"Error: {e}")
        return {"success": False, "reason": str(e), "tools": tool_results}

    while iteration < max_iterations:
        iteration += 1

        try:
            parts = response.candidates[0].content.parts
        except:
            print("\nNo valid response")
            break

        has_function_call = False

        for part in parts:
            if hasattr(part, 'function_call') and part.function_call:
                has_function_call = True
                fn = part.function_call
                tool_name = fn.name
                args = {k: v for k, v in fn.args.items()}

                if tool_name == "apply_discount" and args.get("discount_code") == "":
                    args["discount_code"] = None

                print(f"\n{iteration}. {tool_name}")
                print(f"   Args: {args}")

                try:
                    result = TOOL_FUNCTIONS[tool_name](**args)
                    print(f"   Result: {result}")
                    tool_results[tool_name] = result
                except Exception as e:
                    print(f"   Error: {e}")
                    return {"success": False, "reason": f"Tool error: {e}", "tools": tool_results}

                if tool_name == "check_inventory" and not result.get("available"):
                    print("\nProduct unavailable")
                    return {"success": False, "reason": "Product unavailable", "tools": tool_results}

                if tool_name == "calculate_shipping" and not result.get("available"):
                    print("\nShipping unavailable")
                    return {"success": False, "reason": "Shipping unavailable", "tools": tool_results}

                if tool_name == "process_payment" and not result.get("success"):
                    print("\nPayment failed")
                    return {"success": False, "reason": "Payment failed", "tools": tool_results}


                try:
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
                except Exception as e:
                    print(f"\nLLM response error: {e}")
                    if "send_confirmation_email" in tool_results:
                        break
                    return {"success": False, "reason": f"LLM error: {e}", "tools": tool_results}

            elif hasattr(part, 'text') and part.text:
                print(f"\n{part.text.strip()}")
                has_function_call = False
                break

        if not has_function_call:
            break

        if "send_confirmation_email" in tool_results:
            break

    print("=" * 80)

    required = ["check_inventory", "process_payment", "send_confirmation_email"]
    if not all(t in tool_results for t in required):
        missing = [t for t in required if t not in tool_results]
        return {"success": False, "reason": f"Incomplete: missing {missing}", "tools": tool_results}

    return {"success": True, "tools": tool_results}


if __name__ == "__main__":
    result = ecommerce_agent(
        "I want to order 2 laptops to Jakarta, use discount code WELCOME10, "
        "pay with credit card, email: john@example.com"
    )

    print("\n" + "=" * 80)
    print("FINAL RESULT")
    print("=" * 80)
    print(f"Success: {result['success']}")
    if result["success"]:
        payment = result["tools"].get("process_payment", {})
        print(f"Transaction ID: {payment.get('transaction_id', 'N/A')}")
    print("=" * 80)
