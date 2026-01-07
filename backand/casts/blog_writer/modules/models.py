"""LLM model configuration for Blog Writer cast.

Provides multi-provider LLM support (OpenAI, Anthropic, Google) based on user configuration.
"""

import os
from functools import lru_cache
from typing import Optional

from langchain_core.language_models import BaseChatModel

from casts.blog_writer.modules.state import LLMProvider


def get_llm(
    provider: LLMProvider = LLMProvider.OPENAI,
    model: Optional[str] = None,
    temperature: float = 0.7,
) -> BaseChatModel:
    """Get LLM instance based on provider selection.
    
    Args:
        provider: LLM provider to use
        model: Specific model name (optional, uses defaults)
        temperature: Model temperature setting
        
    Returns:
        Configured LLM instance
    """
    if provider == LLMProvider.OPENAI:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model or "gpt-4o",
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    
    elif provider == LLMProvider.ANTHROPIC:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model or "claude-3-5-sonnet-20241022",
            temperature=temperature,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
    
    elif provider == LLMProvider.GOOGLE:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model or "gemini-1.5-pro",
            temperature=temperature,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
    
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


@lru_cache(maxsize=3)
def get_cached_llm(provider: str, model: Optional[str] = None) -> BaseChatModel:
    """Get cached LLM instance to avoid recreating clients.
    
    Args:
        provider: LLM provider name as string
        model: Specific model name
        
    Returns:
        Cached LLM instance
    """
    return get_llm(LLMProvider(provider), model)
