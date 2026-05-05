#!/usr/bin/env python3
"""
Example script demonstrating how to get reasoning content from GPT-OSS
served via SGLang with --reasoning-parser gpt-oss.

Note:
- `separate_reasoning` is a SGLang extension for its OpenAI-compatible API.
- `reasoning_effort` is optional and may be ignored if the backend version
  does not support it for GPT-OSS yet.

Requires: pip install openai requests

Usage:
    # Non-streaming example
    python scripts/gpt_oss_reasoning_example.py \
        --api-base http://<ip>:8000/v1 \
        --model gpt-oss-120b

    # Streaming example
    python scripts/gpt_oss_reasoning_example.py \
        --api-base http://<ip>:8000/v1 \
        --model gpt-oss-120b \
        --stream
"""
from __future__ import annotations

import argparse
import json

from openai import OpenAI


def demo_non_streaming(client: OpenAI, model: str, question: str, reasoning_effort: str | None):
    """Non-streaming request — reasoning_content returned in one shot."""
    print("=" * 60)
    print("Non-Streaming Request")
    print("=" * 60)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": question}],
        temperature=0.6,
        top_p=0.95,
        max_completion_tokens=4096,
        reasoning_effort=reasoning_effort,
        extra_body={"separate_reasoning": True},
    )

    choice = response.choices[0]
    msg = choice.message
    reasoning = getattr(msg, "reasoning_content", None)
    content = msg.content

    print("\n--- Reasoning ---")
    print(reasoning if reasoning else "(empty)")
    print("\n--- Content ---")
    print(content if content else "(empty)")
    print("\n--- Finish Reason ---")
    print(choice.finish_reason)
    print("\n--- Usage ---")
    print(json.dumps(getattr(response, "usage", None), ensure_ascii=False, indent=2, default=str))
    print()
    return reasoning, content


def demo_streaming(client: OpenAI, model: str, question: str, reasoning_effort: str | None):
    """Streaming request — reasoning and content arrive incrementally."""
    print("=" * 60)
    print("Streaming Request")
    print("=" * 60)

    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": question}],
        temperature=0.6,
        top_p=0.95,
        max_completion_tokens=4096,
        reasoning_effort=reasoning_effort,
        stream=True,
        extra_body={"separate_reasoning": True},
    )

    reasoning_parts: list[str] = []
    content_parts: list[str] = []

    print("\n--- Reasoning (streaming) ---")
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        rc = getattr(delta, "reasoning_content", None)
        if rc:
            reasoning_parts.append(rc)
            print(rc, end="", flush=True)
        if delta.content:
            if reasoning_parts and not content_parts:
                print("\n\n--- Content (streaming) ---")
            content_parts.append(delta.content)
            print(delta.content, end="", flush=True)

    print("\n")
    return "".join(reasoning_parts), "".join(content_parts)


def demo_raw_requests(api_base: str, model: str, question: str, reasoning_effort: str | None):
    """
    Raw requests call — mirrors the terminal snippet used to verify the API.
    """
    import requests

    print("=" * 60)
    print("Raw HTTP Request (requests)")
    print("=" * 60)

    url = f"{api_base}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "temperature": 0.6,
        "top_p": 0.95,
        "max_completion_tokens": 4096,
        "separate_reasoning": True,
    }
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort

    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()

    msg = data["choices"][0]["message"]
    reasoning = msg.get("reasoning_content", "")
    content = msg.get("content", "")

    print("\n--- Reasoning ---")
    print(reasoning if reasoning else "(empty)")
    print("\n--- Content ---")
    print(content if content else "(empty)")
    print("\n--- Finish Reason ---")
    print(data["choices"][0].get("finish_reason"))
    print("\n--- Raw JSON Keys ---")
    print(list(msg.keys()))
    print("\n--- Usage ---")
    print(json.dumps(data.get("usage", {}), ensure_ascii=False, indent=2))
    print()
    return reasoning, content


def demo_disable_reasoning(client: OpenAI, model: str, question: str, reasoning_effort: str | None):
    """
    With separate_reasoning=False, reasoning tokens are merged into content.
    Useful when you want the raw output without separation.
    """
    print("=" * 60)
    print("Reasoning Disabled (merged output)")
    print("=" * 60)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": question}],
        temperature=0.6,
        top_p=0.95,
        max_completion_tokens=4096,
        reasoning_effort=reasoning_effort,
        extra_body={"separate_reasoning": False},
    )

    content = response.choices[0].message.content
    print("\n--- Full Content (reasoning + answer merged) ---")
    print(content if content else "(empty)")
    print()
    return content


def main():
    parser = argparse.ArgumentParser(description="GPT-OSS reasoning demo")
    parser.add_argument("--api-base", required=True, help="e.g. http://10.0.0.1:8000/v1")
    parser.add_argument("--model", default="gpt-oss-120b")
    parser.add_argument(
        "--question",
        default="请详细推导方程 x^2 - 5x + 6 = 0 的解，并最后给出答案。",
    )
    parser.add_argument("--stream", action="store_true", help="Use streaming mode")
    parser.add_argument("--raw", action="store_true", help="Use raw HTTP request instead of OpenAI SDK")
    parser.add_argument("--no-separate", action="store_true", help="Disable reasoning separation")
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        default=None,
        help="Optional reasoning effort. Backend may ignore unsupported values.",
    )
    args = parser.parse_args()

    client = OpenAI(api_key="EMPTY", base_url=args.api_base)

    if args.raw:
        demo_raw_requests(args.api_base, args.model, args.question, args.reasoning_effort)
    elif args.no_separate:
        demo_disable_reasoning(client, args.model, args.question, args.reasoning_effort)
    elif args.stream:
        demo_streaming(client, args.model, args.question, args.reasoning_effort)
    else:
        demo_non_streaming(client, args.model, args.question, args.reasoning_effort)


if __name__ == "__main__":
    main()
