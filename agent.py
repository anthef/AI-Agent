"""E-Commerce Order Processing Agent.

This module implements a simple agentic system for processing e-commerce orders
with multiple dependent tools.

Authors:
    Anthony Edbert Feriyanto

References:
    NONE
"""

from deepeval.tracing import observe
from dotenv import load_dotenv
import re
import json

from llm import call_llm, init_llm
from tools import (
    apply_discount,
    calculate_shipping,
    check_inventory,
    process_payment,
    send_confirmation_email,
)

load_dotenv()

SYSTEM_PROMPT = """You are an e-commerce order processing agent.
Your job is to help customers complete their orders by:
1. Checking product availability
2. Applying discounts if provided
3. Calculating shipping costs
4. Processing payment
5. Sending confirmation email

You must plan your steps carefully and explain your reasoning before taking actions.
Always verify inventory before proceeding with payment.
"""


@observe(type="agent")
def ecommerce_agent(user_input: str) -> dict:
    """Process e-commerce order based on user input.

    Args:
        user_input (str): Customer's order request.

    Returns:
        dict: Complete order processing result including all tool outputs.
    """
    model = init_llm()

    messages = [
        {
        "role": "system", "content": SYSTEM_PROMPT
        }, 
        {
            "role": "user", 
            "content": user_input
        }]

    # Step 1: Reasoning : plan the order processing
    reasoning = call_llm(model, messages)
    print("=" * 80)
    print(reasoning)
    print("=" * 80 + "\n")

    # Step 2: Extract order details 
    order_details = _extract_order_details(user_input)

    print(f"Product ID: {order_details['product_id']}")
    print(f"Quantity: {order_details['quantity']}")
    print(f"Destination: {order_details['destination']}")
    print(f"Discount Code: {order_details.get('discount_code', 'None')}")
    print(f"Payment Method: {order_details['payment_method']}")
    print(f"Customer Email: {order_details['customer_email']}")
    print("=" * 80)

    print("TOOL EXECUTION")
    print("=" * 80)

    # Tool 1: Check inventory
    print("\n1. Checking inventory")
    inventory_result = check_inventory(order_details["product_id"], order_details["quantity"])
    print(f"    Result: {inventory_result}")

    if not inventory_result["available"]:
        return {
            "success": False,
            "reason": "Product not available",
            "inventory": inventory_result,
        }

    # Tool 2: Apply discount (if provided)
    print("\n2. Applying discount")
    discount_result = apply_discount(inventory_result["total_price"], order_details.get("discount_code"))
    print(f"    Result: {discount_result}")

    # Tool 3: Calculate shipping
    print("\n3. Calculating shipping")
    # Estimate weight based on product
    estimated_weight = order_details["quantity"] * 2.5  # kg per item
    shipping_result = calculate_shipping(order_details["destination"], estimated_weight)
    print(f"Result: {shipping_result}")

    if not shipping_result["available"]:
        return {
            "success": False,
            "reason": "Shipping not available",
            "shipping": shipping_result,
        }

    # Calculate final total
    final_total = discount_result["final_price"] + shipping_result["shipping_cost"]

    # Tool 4: Process payment
    print("\n[4] Processing payment...")
    payment_result = process_payment(final_total, order_details["payment_method"])
    print(f"    Result: {payment_result}")

    if not payment_result["success"]:
        return {
            "success": False,
            "reason": "Payment failed",
            "payment": payment_result,
        }

    # Tool 5: Send confirmation email
    print("\n[5] Sending confirmation email")
    order_summary = {
        "transaction_id": payment_result["transaction_id"],
        "product": inventory_result["product_name"],
        "quantity": order_details["quantity"],
        "subtotal": inventory_result["total_price"],
        "discount": discount_result["discount_amount"],
        "shipping": shipping_result["shipping_cost"],
        "total": final_total,
        "delivery_days": shipping_result["estimated_delivery_days"],
    }
    email_result = send_confirmation_email(order_details["customer_email"], order_summary)
    print(f"Result: {email_result}")

    print("=" * 80)

    return {
        "success": True,
        "inventory": inventory_result,
        "discount": discount_result,
        "shipping": shipping_result,
        "payment": payment_result,
        "email": email_result,
        "order_summary": order_summary,
    }


def _extract_order_details(user_input: str) -> dict:
    """Extract order details from user input.

    Uses LLM to extract structured information from natural language input.

    Args:
        user_input (str): User's order request.

    Returns:
        dict: Extracted order details.
    """
    model = init_llm()

    extraction_prompt = f"""Extract the following information from the customer's order request.
    Return ONLY a valid JSON object with these exact keys, no additional text:

    {{
        "product_id": "LAPTOP-001 or MOUSE-002 or KEYBOARD-003 or MONITOR-004 or HEADSET-005",
        "quantity": <number>,
        "destination": "Jakarta or Bandung or Surabaya or Bali or Singapore",
        "discount_code": "WELCOME10 or SAVE50 or VIP20 or null",
        "payment_method": "credit_card or bank_transfer or ewallet",
        "customer_email": "email@example.com"
    }}

    Available products:
    - LAPTOP-001: Business Laptop
    - MOUSE-002: Wireless Mouse
    - KEYBOARD-003: Mechanical Keyboard
    - MONITOR-004: 4K Monitor
    - HEADSET-005: Noise Cancelling Headset

    Available discount codes:
    - WELCOME10: 10% off for new customers
    - SAVE50: $50 off orders over $500
    - VIP20: 20% off for VIP members

    Available destinations:
    - Jakarta, Bandung, Surabaya, Bali, Singapore

    Payment methods:
    - credit_card, bank_transfer, ewallet

    Customer request: "{user_input}"

    Extract the information and return ONLY the JSON object:"""

    messages = [{"role": "user", "content": extraction_prompt}]

    try:
        response = call_llm(model, messages)
        
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()
        
        import json
        details = json.loads(response)
        
        defaults = {
            "product_id": "LAPTOP-001",
            "quantity": 1,
            "destination": "Jakarta",
            "discount_code": None,
            "payment_method": "credit_card",
            "customer_email": "customer@example.com",
        }
        
        for key, default_value in defaults.items():
            if key not in details or details[key] is None or details[key] == "null":
                details[key] = default_value
        
        return details
        
    except Exception as e:
        print(f"Warning: LLM extraction failed ({e}), using fallback pattern matching")
        return _extract_order_details_fallback(user_input)


def _extract_order_details_fallback(user_input: str) -> dict:
    """Fallback extraction using pattern matching.

    Args:
        user_input (str): User's order request.

    Returns:
        dict: Extracted order details.
    """
    details = {
        "product_id": "LAPTOP-001",
        "quantity": 1,
        "destination": "Jakarta",
        "discount_code": None,
        "payment_method": "credit_card",
        "customer_email": "customer@example.com",
    }

    user_input_lower = user_input.lower()

    if "mouse" in user_input_lower:
        details["product_id"] = "MOUSE-002"
    elif "keyboard" in user_input_lower:
        details["product_id"] = "KEYBOARD-003"
    elif "monitor" in user_input_lower:
        details["product_id"] = "MONITOR-004"
    elif "headset" in user_input_lower:
        details["product_id"] = "HEADSET-005"

    for i in range(1, 20):
        if str(i) in user_input:
            details["quantity"] = i
            break

    cities = ["Jakarta", "Bandung", "Surabaya", "Bali", "Singapore"]
    for city in cities:
        if city.lower() in user_input_lower:
            details["destination"] = city
            break

    discount_codes = ["WELCOME10", "SAVE50", "VIP20"]
    for code in discount_codes:
        if code.lower() in user_input_lower:
            details["discount_code"] = code
            break

    if "bank transfer" in user_input_lower or "bank_transfer" in user_input_lower:
        details["payment_method"] = "bank_transfer"
    elif "ewallet" in user_input_lower or "e-wallet" in user_input_lower:
        details["payment_method"] = "ewallet"

    email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", user_input)
    if email_match:
        details["customer_email"] = email_match.group(0)

    return details


if __name__ == "__main__":
    result = ecommerce_agent(
        "I want to order 2 laptops to Jakarta, use discount code WELCOME10, "
        "pay with credit card, email: john@example.com"
    )

    print("\n" + "=" * 80)
    print("FINAL RESULT")
    print("=" * 80)
    print(f"Order Success: {result['success']}")
    if result["success"]:
        print(f"Transaction ID: {result['order_summary']['transaction_id']}")
        print(f"Total Amount: ${result['order_summary']['total']:.2f}")
    print("=" * 80)
