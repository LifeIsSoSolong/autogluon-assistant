import logging
import os
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig
from openai import OpenAI
from langchain_core.messages import AIMessage

from .base_chat import BaseAssistantChat

logger = logging.getLogger(__name__)


class AssistantChatYuanjing(BaseAssistantChat):
    model_name: str
    api_key: str
    base_url: str
    temperature: float = 0.1
    top_p: float = 0.9
    max_tokens: Optional[int] = None
    client: Optional[OpenAI] = None

    def model_post_init(self, __context: Any) -> None:
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.initialize_conversation(self)

    # LangGraph expects the chat object to expose invoke(messages)
    def invoke(self, prompt_messages):  # type: ignore[override]
        if self.client is None:
            raise RuntimeError("OpenAI client not initialized")

        def _normalize_content(content: Any) -> str:
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

        if hasattr(prompt_messages, "to_messages"):
            messages = prompt_messages.to_messages()
        else:
            messages = prompt_messages

        payload: List[Dict[str, str]] = []
        for message in messages:
            role = getattr(message, "type", "user")
            if role == "human":
                role = "user"
            elif role == "ai":
                role = "assistant"
            content = _normalize_content(message.content).strip()
            if not content:
                continue
            payload.append({"role": role, "content": content})

        if not payload:
            combined = "\n".join(
                text for text in (_normalize_content(msg.content) for msg in messages) if text.strip()
            ).strip()
            if not combined:
                raise ValueError("Unable to build non-empty prompt for Yuanjing request.")
            payload.append({"role": "user", "content": combined})

        request_payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": payload,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.max_tokens is not None:
            request_payload["max_tokens"] = self.max_tokens

        response = self.client.chat.completions.create(**request_payload)
        choice = response.choices[0]
        text = getattr(choice.message, "content", "") or ""
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        total_tokens = getattr(usage, "total_tokens", input_tokens + output_tokens) if usage else (input_tokens + output_tokens)
        usage_metadata = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }
        return AIMessage(content=text, usage_metadata=usage_metadata)


def _extract_credentials(config: DictConfig) -> Dict[str, Any]:
    api_key_env = getattr(config, "api_key_env", "YUANJING_API_KEY")
    api_key = os.environ.get(api_key_env) or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(f"Yuanjing-compatible API key not found. Set `{api_key_env}` or `OPENAI_API_KEY`.")

    base_url = getattr(config, "base_url", None) or os.environ.get("YUANJING_BASE_URL")
    if not base_url:
        raise ValueError("Yuanjing-compatible base URL not provided. Configure `base_url` in YAML or set `YUANJING_BASE_URL`.")

    return {"api_key": api_key, "base_url": base_url}


def create_yuanjing_chat(config: DictConfig, session_name: str) -> AssistantChatYuanjing:
    creds = _extract_credentials(config)
    temperature = getattr(config, "temperature", 0.1)
    top_p = getattr(config, "top_p", 0.9)
    max_tokens = getattr(config, "max_tokens", None)

    logger.info(
        "Using Yuanjing OpenAI-compatible model `%s` via `%s` for session `%s`",
        config.model,
        creds["base_url"],
        session_name,
    )

    return AssistantChatYuanjing(
        model_name=config.model,
        api_key=creds["api_key"],
        base_url=creds["base_url"],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        session_name=session_name,
    )


def get_yuanjing_models() -> List[str]:
    return []

