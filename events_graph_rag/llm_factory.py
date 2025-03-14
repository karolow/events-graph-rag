"""
Factory for creating language model instances based on configuration.
"""

from typing import Optional

from langchain_core.language_models.base import BaseLanguageModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from events_graph_rag.config import LLM_CONFIG, logger


class LLMFactory:
    """Factory for creating language model instances."""

    @staticmethod
    def create_llm(
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> BaseLanguageModel:
        """
        Create a language model instance based on configuration.

        Args:
            provider: LLM provider (groq, gemini)
            model_name: Name of the model to use
            temperature: Temperature for generation

        Returns:
            BaseLanguageModel: Language model instance
        """
        # Use configuration values if not provided
        provider = provider or LLM_CONFIG.get("provider", "groq")
        temperature = (
            temperature if temperature is not None else LLM_CONFIG.get("temperature", 0)
        )

        # Create the appropriate LLM based on provider
        if provider.lower() == "groq":
            model_name = model_name or LLM_CONFIG.get(
                "groq_model", "llama-3.3-70b-versatile"
            )
            logger.info(f"Creating Groq LLM with model: {model_name}")
            return ChatGroq(
                model_name=model_name,
                temperature=temperature,
            )
        elif provider.lower() == "gemini":
            model_name = model_name or LLM_CONFIG.get(
                "gemini_model", "gemini-2.0-flash"
            )
            logger.info(f"Creating Gemini LLM with model: {model_name}")
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
