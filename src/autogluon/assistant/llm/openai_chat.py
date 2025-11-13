import logging
import os
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from openai import OpenAI

from .base_chat import BaseAssistantChat

logger = logging.getLogger(__name__)


class AssistantChatOpenAI(ChatOpenAI, BaseAssistantChat):
    """OpenAI chat model with LangGraph support."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initialize_conversation(self)

    def describe(self) -> Dict[str, Any]:
        base_desc = super().describe()
        return {**base_desc, "model": self.model_name, "proxy": self.openai_proxy}


class AssistantChatOpenAIProxy(BaseAssistantChat):
    """OpenAI-compatible chat model using the legacy /chat/completions endpoint."""

    model_name: str
    api_key: str
    base_url: str
    temperature: float = 0.1
    top_p: float = 0.9
    max_tokens: Optional[int] = None
    session_name: str
    client: Optional[OpenAI] = None

    def model_post_init(self, __context: Any) -> None:
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.initialize_conversation(self)

    def describe(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "base_url": getattr(self.client, "base_url", None),
            "mode": "chat.completions",
        }

    def _normalize_content(self, content: Any) -> str:
        if isinstance(content, list):
            parts: List[str] = []
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text") or part.get("content")
                    if text:
                        parts.append(str(text))
                else:
                    parts.append(str(part))
            content = "\n".join(parts)
        elif isinstance(content, dict):
            content = content.get("text") or content.get("content") or ""
        if content is None:
            content = ""
        return str(content)

    def invoke(self, prompt_messages):  # type: ignore[override]
        if hasattr(prompt_messages, "to_messages"):
            messages = prompt_messages.to_messages()
        else:
            messages = prompt_messages

        logger.info(
            "[OpenAIProxy] Invoking model `%s` via `%s` (session=%s, messages=%d)",
            self.model_name,
            self.base_url,
            self.session_name,
            len(messages),
        )

        payload: List[Dict[str, str]] = []
        for message in messages:
            role = getattr(message, "type", "user")
            if role == "human":
                role = "user"
            elif role == "ai":
                role = "assistant"
            content = self._normalize_content(message.content).strip()
            if not content:
                continue
            payload.append({"role": role, "content": content})

        if not payload:
            combined = "\n".join(
                text for text in (self._normalize_content(msg.content) for msg in messages) if text.strip()
            ).strip()
            if not combined:
                raise ValueError("Unable to build non-empty prompt for OpenAI proxy request.")
            payload.append({"role": "user", "content": combined})

        request_payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": payload,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.max_tokens is not None:
            request_payload["max_tokens"] = self.max_tokens

        try:
            response = self.client.chat.completions.create(**request_payload)
        except Exception as exc:
            logger.error(
                "[OpenAIProxy] Request failed for model `%s` via `%s`: %s",
                self.model_name,
                self.base_url,
                exc,
            )
            raise
        choice = response.choices[0]
        text = getattr(choice.message, "content", "") or ""
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        total_tokens = getattr(usage, "total_tokens", input_tokens + output_tokens) if usage else (
            input_tokens + output_tokens
        )
        usage_metadata = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }

        logger.info(
            "[OpenAIProxy] Model `%s` responded (session=%s, input_tokens=%s, output_tokens=%s)",
            self.model_name,
            self.session_name,
            input_tokens,
            output_tokens,
        )
        return AIMessage(content=text, usage_metadata=usage_metadata)


def get_openai_models() -> List[str]:
    try:
        client = OpenAI()
        models = client.models.list()
        return [model.id for model in models if model.id.startswith(("gpt-3.5", "gpt-4", "o1", "o3"))]
    except Exception as e:
        logger.error(f"Error fetching OpenAI models: {e}")
        return []


def create_openai_chat(config, session_name: str) -> AssistantChatOpenAI:
    """Create an OpenAI chat model instance."""
    model = config.model

    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("OpenAI API key not found in environment")

    proxy_url = getattr(config, "proxy_url", None)
    force_proxy_mode = getattr(config, "use_chat_completions_api", False)

    if proxy_url or force_proxy_mode:
        base_url = proxy_url or os.environ.get("OPENAI_API_BASE")
        if not base_url:
            raise ValueError("`proxy_url` must be provided when forcing chat completions API.")

        logger.info(
            "Using OpenAI-compatible proxy via %s for session: %s (model: %s)",
            base_url,
            session_name,
            model,
        )
        return AssistantChatOpenAIProxy(
            model_name=model,
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=base_url,
            temperature=getattr(config, "temperature", 0.1),
            top_p=getattr(config, "top_p", 0.9),
            max_tokens=getattr(config, "max_tokens", None),
            session_name=session_name,
        )

    logger.info(f"Using OpenAI model: {model} for session: {session_name}")
    kwargs = {
        "model_name": model,
        "openai_api_key": os.environ["OPENAI_API_KEY"],
        "session_name": session_name,
        "max_tokens": config.max_tokens,
    }

    if hasattr(config, "temperature"):
        kwargs["temperature"] = config.temperature

    if hasattr(config, "verbose"):
        kwargs["verbose"] = config.verbose

    return AssistantChatOpenAI(**kwargs)
