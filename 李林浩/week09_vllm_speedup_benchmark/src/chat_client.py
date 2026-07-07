"""Single request demo for the vLLM OpenAI-compatible service."""

from __future__ import annotations

import argparse

from openai import OpenAI

from config import DEFAULT_BASE_URL, DEFAULT_MODEL_NAME


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--prompt", default="请解释 vLLM 的核心优势。")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url.rstrip("/"), api_key="EMPTY")
    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": args.prompt}],
        temperature=0.2,
        max_tokens=args.max_tokens,
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
