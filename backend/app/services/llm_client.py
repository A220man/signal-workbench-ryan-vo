"""Opt-in LLM advisory client with OpenAI-compatible, Anthropic, and Gemini adapters.

All LLM calls are server-side only, use bounded timeouts, and never fabricate
successful responses.  Failures are surfaced as explicit error strings so the
calling endpoint can fall back to a grounded (non-LLM) response.
"""
from __future__ import annotations

import json
from typing import Dict, Any, Optional, Tuple

import httpx

from app.core.config import settings


def build_grounded_context(context: Dict[str, Any]) -> Dict[str, Any]:
    """Merge caller-provided context with verified Issue findings."""
    base: Dict[str, Any] = {
        "activation_bug": "Issue #588 confirms ReLU\u00b2 (not SiLU) is correct for BitNet b1.58 FFN. "
        "Measured PPL: 17.1 (ReLU\u00b2) vs 99.8 (SiLU).",
        "build_issues": "Issues #180, #334, #185, #480 document clang/CMake fixes for various platforms.",
        "server_request": "Issue #206 requests persistent in-memory model server for consistent inference.",
        "larger_models": "Issue #8: 70B requires ~17.5 GB, 405B far exceeds consumer RAM even at 1.58-bit.",
    }
    base.update(context)
    return base


def build_prompt(topic: str, context: Dict[str, Any]) -> str:
    """Construct a grounded advisory prompt from the topic and known findings."""
    grounded = build_grounded_context(context)
    prompt = (
        "You are a helpful technical assistant for the Signal Workbench project "
        "(BitNet b1.58 1-bit LLM inference and calibration). "
        "Provide concise, grounded advice. Cite upstream issues when referencing findings. "
        "If you cannot answer from the provided context, state that explicitly and "
        "recommend empirical verification.\n\n"
        f"Topic: {topic}\n\n"
        "Grounded context:\n"
        + json.dumps(grounded, indent=2)
        + "\n\nResponse:"
    )
    return prompt


async def call_openai_compatible(
    prompt: str, api_key: str, base_url: str, model: str
) -> str:
    """Call an OpenAI-compatible chat-completions endpoint (covers Ollama, LiteLLM, OpenRouter)."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.3,
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise ValueError("No choices in LLM response")
        content = choices[0].get("message", {}).get("content", "")
        if not content:
            raise ValueError("Empty content in LLM response")
        return content


async def call_anthropic(prompt: str, api_key: str, model: str) -> str:
    """Call Anthropic messages API."""
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    payload = {
        "model": model,
        "max_tokens": 512,
        "messages": [{"role": "user", "content": prompt}],
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        content = data.get("content", [])
        if not content:
            raise ValueError("No content in Anthropic response")
        text = content[0].get("text", "")
        if not text:
            raise ValueError("Empty content in Anthropic response")
        return text


async def call_gemini(prompt: str, api_key: str, model: str) -> str:
    """Call Gemini generateContent API."""
    url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 512, "temperature": 0.3},
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise ValueError("No candidates in Gemini response")
        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            raise ValueError("Empty content in Gemini response")
        return parts[0].get("text", "")


async def call_llm_advisory(topic: str, context: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Attempt an LLM advisory call through the configured provider.

    Returns:
        (content, None) on success.
        (None, error_message) when the call fails or LLM_BASE_URL is not configured.
    """
    if not settings.LLM_API_KEY:
        return None, "LLM_API_KEY not configured on server"

    if not settings.LLM_BASE_URL:
        # No endpoint configured — fall back to grounded response without network call.
        return None, "LLM_BASE_URL not configured; returning grounded response only"

    prompt = build_prompt(topic, context)
    provider = settings.LLM_PROVIDER
    model = settings.LLM_MODEL
    api_key = settings.LLM_API_KEY
    base_url = settings.LLM_BASE_URL

    try:
        if provider == "anthropic":
            content = await call_anthropic(prompt, api_key, model)
        elif provider == "gemini":
            content = await call_gemini(prompt, api_key, model)
        else:
            # openai-compatible (default — covers Ollama, LiteLLM, OpenRouter, OpenAI)
            content = await call_openai_compatible(prompt, api_key, base_url, model)
        return content, None
    except httpx.HTTPStatusError as exc:
        return None, f"LLM provider error: HTTP {exc.response.status_code}"
    except httpx.TimeoutException:
        return None, "LLM provider timed out after 30s"
    except Exception as exc:
        return None, f"LLM provider error: {type(exc).__name__}: {exc}"