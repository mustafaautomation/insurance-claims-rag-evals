"""Chat providers. The fake provider is deterministic and follows the prompt rules."""

from typing import Protocol

from harborline.prompts import (
    AFFIRM,
    HANDOFF_RESPONSE,
    OVERTURN,
    REFUSAL_RESPONSE,
    UNKNOWN_SENTENCE,
    guardrails_enabled,
    is_approval_question,
    is_dispute_question,
    is_out_of_scope_topic,
    minimum_score_from_prompt,
    parse_user_message,
)


class ChatProvider(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class FakeProvider:
    """Rule follower for offline runs and tests. It is not a language model."""

    def complete(self, system: str, user: str) -> str:
        question, pages = parse_user_message(user)
        top_text = pages[0].text if pages else ""
        top_score = pages[0].score if pages else 0.0
        if not guardrails_enabled(system):
            if is_approval_question(question):
                return AFFIRM
            if is_dispute_question(question):
                return OVERTURN
            return top_text
        if is_approval_question(question):
            return REFUSAL_RESPONSE
        if is_dispute_question(question):
            return HANDOFF_RESPONSE
        if is_out_of_scope_topic(question):
            return UNKNOWN_SENTENCE
        minimum = minimum_score_from_prompt(system)
        if minimum is None:
            raise ValueError("guardrail prompt is missing the minimum retrieval score")
        if not pages or top_score < minimum:
            return UNKNOWN_SENTENCE
        return top_text


class OpenAIProvider:
    def __init__(self, model: str) -> None:
        self.model = model

    def complete(self, system: str, user: str) -> str:
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = response.choices[0].message.content
        if isinstance(content, str):
            return content
        return ""


class AnthropicProvider:
    def __init__(self, model: str = "claude-sonnet-4-5") -> None:
        self.model = model

    def complete(self, system: str, user: str) -> str:
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "AnthropicProvider needs the anthropic package. Install the anthropic extra."
            ) from exc
        client = anthropic.Anthropic()
        message = client.messages.create(
            model=self.model,
            max_tokens=800,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        parts: list[str] = []
        for block in message.content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)
