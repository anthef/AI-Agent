"""E-Commerce Order Processing Tools.

This module contains hardcoded tools for simulating e-commerce operations.
All tools are deterministic and do not connect to real APIs.

Authors:
    Anthony Edbert Feriyanto

References:
    NONE
"""

from deepeval.tracing import observe
import random
import string


@observe(type="tool")
def check_inventory(product_id: str, quantity: int) -> dict:
    """Check product inventory availability.

    Args:
        product_id (str): The product identifier.
        quantity (int): Requested quantity.

    Returns:
        dict: Inventory status with availability and stock information.
    """
    inventory = {
        "LAPTOP-001": {
            "name": "Business Laptop", 
            "stock": 50, 
            "price": 1200
            },
        "MOUSE-002": {
            "name": "Wireless Mouse", 
            "stock": 100, 
            "price": 25
            },
        "KEYBOARD-003": {
            "name": "Mechanical Keyboard", 
            "stock": 30, 
            "price": 80
            },
        "MONITOR-004": {
            "name": "4K Monitor", 
            "stock": 20, 
            "price": 450
            },
        "HEADSET-005": {
            "name": "Noise Cancelling Headset", 
            "stock": 15, 
            "price": 150
            },
    }

    if product_id not in inventory:
        return {
            "product_id": product_id,
            "available": False,
            "reason": "Product not found",
            "stock": 0,
        }

    product = inventory[product_id]
    available = product["stock"] >= quantity

    return {
        "product_id": product_id,
        "product_name": product["name"],
        "available": available,
        "requested_quantity": quantity,
        "stock": product["stock"],
        "unit_price": product["price"],
        "total_price": product["price"] * quantity if available else 0,
    }


@observe(type="tool")
def apply_discount(total_price: float, discount_code: str | None = None) -> dict:
    """Apply discount code to order total.

    Args:
        total_price (float): Original total price.
        discount_code (str | None, optional): Discount code to apply. Defaults to None.

    Returns:
        dict: Discount information and final price.
    """
    discounts = {
        "WELCOME10": {
            "type": "percentage", 
            "value": 10, 
            "description": "10% off for new customers"
            },
        "SAVE50": {
            "type": "fixed", 
            "value": 50, 
            "description": "$50 off orders over $500"
            },
        "VIP20": {
            "type": "percentage", 
            "value": 20, 
            "description": "20% off for VIP members"
            },
    }

    if not discount_code or discount_code not in discounts:
        return {
            "discount_applied": False,
            "discount_code": discount_code,
            "original_price": total_price,
            "discount_amount": 0,
            "final_price": total_price,
        }

    discount = discounts[discount_code]

    if discount["type"] == "percentage":
        discount_amount = total_price * (discount["value"] / 100)
    else:  
        discount_amount = min(discount["value"], total_price)

    final_price = total_price - discount_amount

    return {
        "discount_applied": True,
        "discount_code": discount_code,
        "discount_description": discount["description"],
        "original_price": total_price,
        "discount_amount": discount_amount,
        "final_price": final_price,
    }


@observe(type="tool")
def calculate_shipping(destination_city: str, total_weight_kg: float) -> dict:
    """Calculate shipping cost based on destination and weight.

    Args:
        destination_city (str): Destination city name.
        total_weight_kg (float): Total package weight in kilograms.

    Returns:
        dict: Shipping information including cost and estimated delivery.
    """
    shipping_zones = {
        "Jakarta": {
            "base_cost": 10, 
            "per_kg": 2, 
            "delivery_days": 1
            },
        "Bandung": {
            "base_cost": 15, 
            "per_kg": 3, 
            "delivery_days": 2
            },
        "Surabaya": {
            "base_cost": 20, 
            "per_kg": 4, 
            "delivery_days": 3
            },
        "Bali": {
            "base_cost": 30, 
            "per_kg": 5, 
            "delivery_days": 4
            },
        "Singapore": {
            "base_cost": 50, 
            "per_kg": 8, 
            "delivery_days": 5
            },
    }

    if destination_city not in shipping_zones:
        return {
            "destination": destination_city,
            "available": False,
            "reason": "Shipping not available to this location",
            "shipping_cost": 0,
        }

    zone = shipping_zones[destination_city]
    shipping_cost = zone["base_cost"] + (total_weight_kg * zone["per_kg"])

    return {
        "destination": destination_city,
        "available": True,
        "weight_kg": total_weight_kg,
        "shipping_cost": shipping_cost,
        "estimated_delivery_days": zone["delivery_days"],
    }


@observe(type="tool")
def process_payment(amount: float, payment_method: str) -> dict:
    """Process payment for the order.

    Args:
        amount (float): Total amount to charge.
        payment_method (str): Payment method (credit_card, bank_transfer, ewallet).

    Returns:
        dict: Payment processing result.
    """
    valid_methods = ["credit_card", "bank_transfer", "ewallet"]

    if payment_method not in valid_methods:
        return {
            "success": False,
            "payment_method": payment_method,
            "reason": f"Invalid payment method. Valid methods: {', '.join(valid_methods)}",
            "transaction_id": None,
        }

    transaction_id = "TXN-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=10))

    return {
        "success": True,
        "payment_method": payment_method,
        "amount": amount,
        "transaction_id": transaction_id,
        "status": "completed",
    }


@observe(type="tool")
def send_confirmation_email(customer_email: str, order_summary: dict) -> dict:
    """Send order confirmation email to customer.

    Args:
        customer_email (str): Customer's email address.
        order_summary (dict): Complete order information.

    Returns:
        dict: Email sending status.
    """
    return {
        "email_sent": True,
        "recipient": customer_email,
        "subject": f"Order Confirmation - {order_summary.get('transaction_id', 'N/A')}",
        "order_summary": order_summary,
        "status": "delivered",
    }
