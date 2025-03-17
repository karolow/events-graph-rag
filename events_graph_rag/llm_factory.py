"""
Factory for creating language model instances based on configuration.
"""

from typing import Dict, List, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.base import BaseLanguageModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from events_graph_rag.config import LLM_CONFIG, logger


class LLMFactory:
    """Factory for creating language model instances."""

    @staticmethod
    def create_llm(
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        is_cypher: bool = False,
    ) -> BaseLanguageModel:
        """
        Create a language model instance based on configuration.

        Args:
            provider: LLM provider (groq, gemini, openai, anthropic)
            model_name: Name of the model to use
            temperature: Temperature for generation
            is_cypher: Whether this LLM is for Cypher generation

        Returns:
            BaseLanguageModel: Language model instance
        """
        # Use Cypher-specific configuration if requested
        config = LLM_CONFIG["cypher"] if is_cypher else LLM_CONFIG

        # Use configuration values if not provided
        provider = provider or config.get("provider", "groq")
        temperature = (
            temperature if temperature is not None else config.get("temperature", 0)
        )

        # Create the appropriate LLM based on provider
        if provider.lower() == "groq":
            model_name = model_name or config.get(
                "groq_model", "llama-3.3-70b-versatile"
            )
            logger.info(f"Creating Groq LLM with model: {model_name}")
            return ChatGroq(
                model_name=model_name,
                temperature=temperature,
            )
        elif provider.lower() == "gemini":
            model_name = model_name or config.get("gemini_model", "gemini-2.0-flash")
            logger.info(f"Creating Gemini LLM with model: {model_name}")
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
            )
        elif provider.lower() == "openai":
            model_name = model_name or config.get("openai_model", "gpt-4o")
            logger.info(f"Creating OpenAI LLM with model: {model_name}")
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
            )
        elif provider.lower() == "anthropic":
            model_name = model_name or config.get(
                "anthropic_model", "claude-3.7-sonnet"
            )
            logger.info(f"Creating Anthropic LLM with model: {model_name}")
            return ChatAnthropic(
                model=model_name,
                temperature=temperature,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    @staticmethod
    def create_cypher_llm(
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> BaseLanguageModel:
        """
        Create a language model instance specifically for Cypher generation.

        Args:
            provider: LLM provider (groq, gemini, openai, anthropic)
            model_name: Name of the model to use
            temperature: Temperature for generation

        Returns:
            BaseLanguageModel: Language model instance
        """
        return LLMFactory.create_llm(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            is_cypher=True,
        )

    @staticmethod
    def get_available_models(provider: str) -> List[str]:
        """
        Get the list of available models for a provider.

        Args:
            provider: LLM provider (groq, gemini, openai, anthropic)

        Returns:
            List[str]: List of available model names
        """
        available_models = LLM_CONFIG.get("available_models", {})
        return available_models.get(provider.lower(), [])

    @staticmethod
    def get_provider_for_model(model_name: str) -> Optional[str]:
        """
        Get the provider for a given model name.

        Args:
            model_name: Name of the model

        Returns:
            Optional[str]: Provider name or None if not found
        """
        available_models = LLM_CONFIG.get("available_models", {})
        for provider, models in available_models.items():
            if model_name in models:
                return provider
        return None
