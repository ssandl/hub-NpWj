"""Health check for an OpenAI-compatible vLLM service."""

from __future__ import annotations

import argparse
import sys

import httpx

from config import DEFAULT_BASE_URL


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    args = parser.parse_args()

    url = args.base_url.rstrip("/") + "/models"
    try:
        response = httpx.get(url, timeout=10.0)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] vLLM service is not ready: {exc}")
        return 1

    print("[OK] vLLM service is ready.")
    print(response.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
