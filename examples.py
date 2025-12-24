"""Example scenarios for testing the e-commerce agent.

This module contains three different scenarios to demonstrate the agent's capabilities.

Authors:
    Anthony Edbert Feriyanto

References:
    NONE
"""

from dotenv import load_dotenv

from agent import ecommerce_agent

load_dotenv()


def scenario_1_standard_order():
    """Scenario 1: Standard order with discount code."""
    print("\n" + "=" * 80)
    print("SCENARIO 1: Standard Order with Discount")
    print("=" * 80)

    user_input = (
        "I want to order 2 laptops to Jakarta, "
        "use discount code WELCOME10, "
        "pay with credit card, "
        "email: john.doe@example.com"
    )

    print(f"\nUser Input: {user_input}\n")
    result = ecommerce_agent(user_input)

    print("\n" + "=" * 80)
    print("SCENARIO 1 RESULT")
    print("=" * 80)
    print(f"Success: {result['success']}")
    if result["success"]:
        summary = result["order_summary"]
        print(f"Transaction ID: {summary['transaction_id']}")
        print(f"Product: {summary['product']} x {summary['quantity']}")
        print(f"Subtotal: ${summary['subtotal']:.2f}")
        print(f"Discount: -${summary['discount']:.2f}")
        print(f"Shipping: ${summary['shipping']:.2f}")
        print(f"Total: ${summary['total']:.2f}")
        print(f"Estimated Delivery: {summary['delivery_days']} days")
    print("=" * 80)


def scenario_2_vip_order():
    """Scenario 2: VIP order with international shipping."""
    print("\n" + "=" * 80)
    print("SCENARIO 2: VIP Order with International Shipping")
    print("=" * 80)

    user_input = (
        "Order 1 monitor and ship to Singapore, "
        "I have VIP20 discount code, "
        "payment via ewallet, "
        "send confirmation to vip.customer@company.com"
    )

    print(f"\nUser Input: {user_input}\n")
    result = ecommerce_agent(user_input)

    print("\n" + "=" * 80)
    print("SCENARIO 2 RESULT")
    print("=" * 80)
    print(f"Success: {result['success']}")
    if result["success"]:
        summary = result["order_summary"]
        print(f"Transaction ID: {summary['transaction_id']}")
        print(f"Product: {summary['product']} x {summary['quantity']}")
        print(f"Subtotal: ${summary['subtotal']:.2f}")
        print(f"Discount: -${summary['discount']:.2f}")
        print(f"Shipping: ${summary['shipping']:.2f}")
        print(f"Total: ${summary['total']:.2f}")
        print(f"Estimated Delivery: {summary['delivery_days']} days")
    print("=" * 80)


def scenario_3_bulk_order():
    """Scenario 3: Bulk order with large discount."""
    print("\n" + "=" * 80)
    print("SCENARIO 3: Bulk Order with Large Discount")
    print("=" * 80)

    user_input = (
        "I need 5 headsets delivered to Surabaya, "
        "apply SAVE50 discount, "
        "pay by bank transfer, "
        "email: procurement@company.id"
    )

    print(f"\nUser Input: {user_input}\n")
    result = ecommerce_agent(user_input)

    print("\n" + "=" * 80)
    print("SCENARIO 3 RESULT")
    print("=" * 80)
    print(f"Success: {result['success']}")
    if result["success"]:
        summary = result["order_summary"]
        print(f"Transaction ID: {summary['transaction_id']}")
        print(f"Product: {summary['product']} x {summary['quantity']}")
        print(f"Subtotal: ${summary['subtotal']:.2f}")
        print(f"Discount: -${summary['discount']:.2f}")
        print(f"Shipping: ${summary['shipping']:.2f}")
        print(f"Total: ${summary['total']:.2f}")
        print(f"Estimated Delivery: {summary['delivery_days']} days")
    print("=" * 80)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("E-COMMERCE AGENT - EXAMPLE SCENARIOS")
    print("=" * 80)

    scenario_1_standard_order()
    print("\n\n")

    scenario_2_vip_order()
    print("\n\n")

    scenario_3_bulk_order()

    print("\n" + "=" * 80)
    print("ALL SCENARIOS COMPLETED")
    print("=" * 80)
