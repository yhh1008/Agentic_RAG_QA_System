from __future__ import annotations

from typing import Type, TypeVar

from pydantic import BaseModel

from src.config.settings import settings


T = TypeVar("T", bound=BaseModel)


def get_chat_model():
    from langchain_deepseek import ChatDeepSeek

    return ChatDeepSeek(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=0,
    )


def invoke_structured(messages: list[dict], schema: Type[T]) -> T:
    model = get_chat_model().with_structured_output(schema)
    return model.invoke(messages)
