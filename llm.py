"""LLM Wrapper for Gemini 2.5 Flash.

This module provides a simple wrapper for Google's Gemini API.

Authors:
    Anthony Edbert Feriyanto

References:
    NONE
"""

import os
from typing import Any

import google.generativeai as genai
from deepeval.tracing import observe
# from deepeval.metrics import ToolCorrectnessMetric, ArgumentCorrectnessMetric

# Metrics commented out - requires OpenAI API key
# tool_correctness = ToolCorrectnessMetric(threshold=0.7, include_reason=True)
# argument_correctness = ArgumentCorrectnessMetric(threshold=0.7, include_reason=True)

def init_llm() -> Any:
    """Initialize Gemini LLM model.

    Returns:
        Any: Configured Gemini GenerativeModel instance.

    Raises:
        ValueError: If GEMINI_API_KEY is not set in environment.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")

    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")



# Metrics commented out - uncomment when OpenAI API key is available
# @observe(type="llm", metrics=[tool_correctness, argument_correctness])
@observe(type="llm")
def call_llm(model: Any, messages: list[dict]) -> str:
    """Call LLM with formatted messages.

    This function has metrics attached for evaluating tool selection
    and argument generation (action layer).

    Args:
        model (Any): Gemini model instance.
        messages (list[dict]): List of message dictionaries with 'role' and 'content' keys.

    Returns:
        str: Generated response text.
    """
    prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    response = model.generate_content(prompt)
    return response.text
