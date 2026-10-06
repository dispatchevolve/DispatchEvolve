# Adapted for DispatchEvolve; see THIRD_PARTY_NOTICES.md for origin and changes.
"""
Adapted from SakanaAI/ShinkaEvolve (Apache-2.0 License)
Original source: https://github.com/SakanaAI/ShinkaEvolve/blob/main/shinka/llm/embedding.py
"""

import os
import litellm
from typing import Union, List
import logging

logger = logging.getLogger(__name__)

M = 1_000_000

OPENAI_EMBEDDING_MODELS = [
    "text-embedding-3-small",
    "text-embedding-3-large",
]

AZURE_EMBEDDING_MODELS = [
    "azure-text-embedding-3-small",
    "azure-text-embedding-3-large",
]

OPENAI_EMBEDDING_COSTS = {
    "text-embedding-3-small": 0.02 / M,
    "text-embedding-3-large": 0.13 / M,
}


class EmbeddingClient:
    def __init__(self, model_name: str = "text-embedding-3-small"):
        """
        Initialize the EmbeddingClient.

        Args:
            model (str): The OpenAI embedding model name to use.
        """
        self.model_name = model_name
        self._litellm_model, self._api_key, self._api_base = self._resolve(model_name)

    def _resolve(self, model_name: str) -> tuple[str, str | None, str | None]:
        if model_name in OPENAI_EMBEDDING_MODELS:
            api_key = os.getenv("OPENAI_EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
            return model_name, api_key, None
        elif model_name in AZURE_EMBEDDING_MODELS:
            deployment = model_name.split("azure-")[-1]
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            api_base = os.getenv("AZURE_API_ENDPOINT")
            return f"azure/{deployment}", api_key, api_base
        else:
            raise ValueError(f"Invalid embedding model: {model_name}")

    def get_embedding(self, code: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Computes the text embedding for a code string.

        Args:
            code (str, list[str]): The code as a string or list
                of strings.

        Returns:
            list: Embedding vector for the code or None if an error
                occurs.
        """
        if isinstance(code, str):
            code = [code]
            single_code = True
        else:
            single_code = False
        try:
            kwargs = {"model": self._litellm_model, "input": code, "encoding_format": "float"}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._api_base:
                kwargs["api_base"] = self._api_base
            response = litellm.embedding(**kwargs)
            if single_code:
                return response.data[0]["embedding"]
            else:
                return [d["embedding"] for d in response.data]
        except Exception as e:
            logger.info(f"Error getting embedding: {e}")
            if single_code:
                return [], 0.0
            else:
                return [[]], 0.0
