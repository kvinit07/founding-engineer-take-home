"""Two small, explicit official-SDK adapters. The harness owns every tool call.

References (checked November 2026):
https://googleapis.github.io/python-genai/ (manual function calling and Content)
https://aistudio.google.com/docs/models/gemini-3.8-flash (stable tool-capable model)
https://console.groq.com/docs/tool-use/local-tool-calling
https://console.groq.com/docs/models

Storage is provider-neutral. Gemini's native Content JSON is retained separately,
because signatures must survive byte-for-byte. Completed calls from other providers
are imported to Gemini as explicit JSON transcript text: they are historical facts,
not invented signed Gemini calls. Groq accepts both origins as structured messages.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import time
import uuid
from typing import Any, Callable


DEFAULT_MODELS = {"gemini": "gemini-3.8-flash", "groq": "openai/gpt-oss-20b"}


class ProviderError(Exception):
    """An actionable provider error safe to show without credentials or headers."""


def _load_env() -> None:
    """Load keys inside this disposable worker; explicit process env takes priority."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        raise ProviderError("python-dotenv missing. Run: python -m pip install -r requirements.txt") from None
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def to_groq_messages(messages: list[dict], system_prompt: str) -> list[dict]:
    """Convert saved neutral history to Groq's Chat Completions wire format."""
    converted = [{"role": "system", "content": system_prompt}]
    for message in messages:
        role = message["role"]
        if role == "tool":
            converted.append({
                "role": "tool",
                "tool_call_id": message["tool_call_id"],
                "content": _json(message["content"]),
            })
        elif role == "assistant":
            item = {"role": "assistant", "content": message.get("content") or None}
            if message.get("tool_calls"):
                item["tool_calls"] = [{
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": (
                            call["arguments"] if isinstance(call["arguments"], str)
                            else _json(call["arguments"])
                        ),
                    },
                } for call in message["tool_calls"]]
            else:
                item["content"] = message.get("content", "")
            converted.append(item)
        elif role == "user":
            converted.append({"role": "user", "content": message["content"]})
        else:
            raise ProviderError(f"Unsupported saved message role: {role!r}.")
    return converted


def to_gemini_contents(messages: list[dict], model: str | None = None) -> list[dict]:
    """Replay native Gemini parts; preserve foreign calls/results as JSON text.

    Several results for one Gemini parallel-call turn must occupy a single user
    Content. We only combine consecutive user parts; native model Content stays
    intact, including hidden thought parts and their base64-encoded signatures.
    """
    converted: list[dict] = []
    native_calls: dict[str, str | None] = {}

    def append(role: str, parts: list[dict]) -> None:
        if role == "user" and converted and converted[-1]["role"] == "user":
            converted[-1]["parts"].extend(parts)
        else:
            converted.append({"role": role, "parts": parts})

    for message in messages:
        role = message["role"]
        if role == "user":
            append("user", [{"text": message["content"]}])
        elif role == "assistant":
            native = message.get("native") or {}
            if (
                message.get("provider") == "gemini"
                and native.get("content")
                and (model is None or native.get("model") == model)
            ):
                content = copy.deepcopy(native["content"])
                converted.append(content)
                raw_calls = [p["function_call"] for p in content.get("parts", [])
                             if p.get("function_call")]
                for call, raw in zip(message.get("tool_calls", []), raw_calls):
                    native_calls[call["id"]] = raw.get("id")
            else:
                text = message.get("content", "")
                if message.get("tool_calls"):
                    text += "\nHistorical tool requests (already handled by the application):\n"
                    text += _json(message["tool_calls"])
                append("model", [{"text": text or "[No text in this historical turn.]"}])
        elif role == "tool":
            call_id = message["tool_call_id"]
            if call_id in native_calls:
                response = {"name": message["name"], "response": message["content"]}
                if native_calls[call_id]:
                    response["id"] = native_calls[call_id]
                append("user", [{"function_response": response}])
            else:
                append("user", [{"text": "Historical tool result:\n" + _json({
                    "tool_call_id": call_id,
                    "name": message["name"],
                    "result": message["content"],
                })}])
        else:
            raise ProviderError(f"Unsupported saved message role: {role!r}.")
    return converted


def _status(error: Exception) -> int | None:
    value = getattr(error, "status_code", None) or getattr(error, "code", None)
    return value if isinstance(value, int) else None


def _request(action: Callable[[], Any], provider: str) -> Any:
    """Retry only transient model requests, never application tool side effects."""
    for attempt in range(3):
        try:
            return action()
        except Exception as error:
            code = _status(error)
            if (code == 429 or code is not None and 500 <= code <= 599) and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            if code in (401, 403):
                detail = "authentication/permission failed; check its API key and account access"
            elif code == 404:
                detail = "model unavailable; check the configured model and account access"
            elif code == 429:
                detail = "rate limit/quota reached after retries; wait or switch provider"
            elif code is not None and code >= 500:
                detail = "service unavailable after retries; try again or switch provider"
            elif "timeout" in type(error).__name__.lower():
                detail = "request timed out; use /resume or switch provider"
            elif "connection" in type(error).__name__.lower():
                detail = "connection failed; check your internet/proxy configuration"
            else:
                detail = f"request failed ({type(error).__name__}); check model configuration"
            # SDK exception strings may contain request headers, URLs or key values.
            raise ProviderError(f"{provider}: {detail}.") from None
    raise AssertionError("unreachable")


def _gemini(model: str, messages: list[dict], tools: list[dict], prompt: str, key: str) -> dict:
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ProviderError("Gemini SDK missing. Run: python -m pip install -r requirements.txt") from None

    # JSON validation decodes the SDK's base64 signature representation back to
    # bytes correctly; model_validate(dict) can interpret those strings as UTF-8.
    contents = [types.Content.model_validate_json(_json(item))
                for item in to_gemini_contents(messages, model)]
    declarations = [types.FunctionDeclaration(
        name=tool["name"], description=tool["description"],
        parameters_json_schema=tool["parameters"],
    ) for tool in tools]
    config = types.GenerateContentConfig(
        system_instruction=prompt,
        tools=[types.Tool(function_declarations=declarations)] if declarations else None,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        temperature=0.1,
        max_output_tokens=4096,
    )
    # SDK retries are disabled so there is exactly one bounded retry policy.
    with genai.Client(api_key=key, http_options=types.HttpOptions(
        timeout=45_000, retry_options=types.HttpRetryOptions(attempts=1),
    )) as client:
        response = _request(lambda: client.models.generate_content(
            model=model, contents=contents, config=config,
        ), "gemini")
    content = response.candidates[0].content if response.candidates else None
    if content is None or not content.parts:
        raise ProviderError("gemini returned no usable content; use /resume or another model.")
    calls = []
    call_prefix = uuid.uuid4().hex[:16]
    for part in content.parts:
        if part.function_call:
            call = part.function_call
            calls.append({
                # Provider IDs can repeat across turns (some models return "0").
                # Native Gemini IDs remain in native Content for replay; the
                # application ID is unique across the full saved conversation.
                "id": f"call_{call_prefix}_{len(calls)}",
                "name": call.name or "",
                "arguments": dict(call.args or {}),
            })
    text = "\n".join(p.text for p in content.parts if p.text and not p.thought)
    if not text and not calls:
        raise ProviderError("gemini returned no text or tool calls; use /resume or another model.")
    return {
        "role": "assistant", "content": text, "tool_calls": calls, "provider": "gemini",
        "native": {"model": model, "content": json.loads(content.model_dump_json(exclude_none=True))},
    }


def _groq(model: str, messages: list[dict], tools: list[dict], prompt: str, key: str) -> dict:
    try:
        from groq import Groq
    except ImportError:
        raise ProviderError("Groq SDK missing. Run: python -m pip install -r requirements.txt") from None

    with Groq(api_key=key, timeout=45.0, max_retries=0) as client:
        response = _request(lambda: client.chat.completions.create(
            model=model,
            messages=to_groq_messages(messages, prompt),
            tools=[{"type": "function", "function": tool} for tool in tools],
            tool_choice="auto",
            temperature=0.1,
            max_completion_tokens=4096,
        ), "groq")
    if not response.choices:
        raise ProviderError("groq returned no choices; use /resume or another model.")
    message = response.choices[0].message
    calls = []
    call_prefix = uuid.uuid4().hex[:16]
    for call in message.tool_calls or []:
        arguments = call.function.arguments
        try:
            arguments = json.loads(arguments)
        except (TypeError, ValueError):
            pass  # Preserve malformed args so the harness records a tool error.
        calls.append({"id": f"call_{call_prefix}_{len(calls)}", "name": call.function.name, "arguments": arguments})
    if not message.content and not calls:
        raise ProviderError("groq returned no text or tool calls; use /resume or another model.")
    return {"role": "assistant", "content": message.content or "", "tool_calls": calls, "provider": "groq"}


def generate(provider: str, model: str, messages: list[dict], tools: list[dict], system_prompt: str) -> dict:
    """Perform one model turn, without executing tools or mutating saved history."""
    if provider not in DEFAULT_MODELS:
        raise ProviderError("Choose one of the supported providers: gemini, groq.")
    _load_env()
    variable = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY"}[provider]
    key = os.environ.get(variable, "").strip()
    if not key or key.lower() in {"...", "your-api-key", "your-key-here"}:
        raise ProviderError(f"{variable} is missing. Set it in .env or your environment, then use /resume.")
    if not isinstance(model, str) or not model.strip():
        raise ProviderError(f"No model configured for {provider}.")
    return {"gemini": _gemini, "groq": _groq}[provider](model, messages, tools, system_prompt, key)
