"""DeepEval Evaluator Wrapper for Gemini.

This module provides a wrapper class to use Google's Gemini models
as a custom judge/evaluator for DeepEval metrics.

Authors:
    Anthony Edbert Feriyanto

References:
    NONE
"""

import os
# import json
import re
# from typing import Optional
from google import genai
from deepeval.models.base_model import DeepEvalBaseLLM

class GeminiEvaluator(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "gemini-2.5-flash-lite"):
        self.model_name = model_name
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        model_id = self.model_name if self.model_name.startswith("models/") else f"models/{self.model_name}"
        response = self.client.models.generate_content(
            model=model_id,
            contents=prompt
        )
        return self._clean_json_response(response.text)

    async def a_generate(self, prompt: str) -> str:
        model_id = self.model_name if self.model_name.startswith("models/") else f"models/{self.model_name}"
        response = await self.client.aio.models.generate_content(
            model=model_id,
            contents=prompt
        )
        return self._clean_json_response(response.text)

    def get_model_name(self):
        return self.model_name

    def _clean_json_response(self, text: str) -> str:
        text = re.sub(r"```json\s?|```", "", text).strip()
        return text