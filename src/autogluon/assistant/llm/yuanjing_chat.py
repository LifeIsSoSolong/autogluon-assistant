import logging
import os
from typing import Any, Dict, List

from omegaconf import DictConfig

from .openai_chat import AssistantChatOpenAI

logger = logging.getLogger(__name__)


def _build_kwargs(config: DictConfig, session_name: str) -> Dict[str, Any]:
    """Build keyword arguments for AssistantChatOpenAI when using a Yuanjing-compatible endpoint."""
    api_key_env = getattr(config, "api_key_env", "YUANJING_API_KEY")
    api_key = os.environ.get(api_key_env) or os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            f"Yuanjing-compatible API key not found. Set `{api_key_env}` or `OPENAI_API_KEY`."
        )

    base_url = getattr(config, "base_url", None) or os.environ.get("YUANJING_BASE_URL")
    if not base_url:
        raise ValueError(
            "Yuanjing-compatible base URL not provided. Configure `base_url` in the YAML or set `YUANJING_BASE_URL`."
        )

    kwargs: Dict[str, Any] = {
        "model_name": config.model,
        "openai_api_key": api_key,
        "session_name": session_name,
        "max_tokens": getattr(config, "max_tokens", None),
        "openai_api_base": base_url,
    }

    if hasattr(config, "temperature"):
        kwargs["temperature"] = config.temperature
    if hasattr(config, "top_p"):
        kwargs["top_p"] = config.top_p
    if hasattr(config, "verbose"):
        kwargs["verbose"] = config.verbose

    kwargs["openai_proxy"] = base_url

    return kwargs


def create_yuanjing_chat(config: DictConfig, session_name: str) -> AssistantChatOpenAI:
    """Create an AssistantChatOpenAI instance configured for a Yuanjing OpenAI-compatible endpoint."""
    kwargs = _build_kwargs(config, session_name)
    logger.info(
        "Using Yuanjing OpenAI-compatible model `%s` via `%s` for session `%s`",
        kwargs["model_name"],
        kwargs["openai_api_base"],
        session_name,
    )
    return AssistantChatOpenAI(**kwargs)


def get_yuanjing_models() -> List[str]:
    """Yuanjing services may not expose `/models`; rely on user-specified model IDs."""
    return []
