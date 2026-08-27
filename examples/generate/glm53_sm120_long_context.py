# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate near-limit GLM-5.3 retrieval through an OpenAI-compatible API."""

import argparse
import json
import time
import urllib.request

MODEL = "GLM-5.3-Flash"
NEEDLE = "RIVENDELL-262K-PASS"
TARGET_TOKENS = 240_000
EXPECTED_PROMPT_TOKENS = 239_994
EXPECTED_MAX_MODEL_LEN = 262_144
UNIT = "Neutral filler sentence for validating a long context window. "
PREFIX = (
    "Long-context memory test. The verification code is "
    f"{NEEDLE}. Remember it exactly.\n\n"
)
SUFFIX = (
    "\n\nWhat was the verification code near the beginning? "
    "Return only that code and nothing else."
)


def post(base_url: str, path: str, payload: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def count_tokens(base_url: str, repeats: int) -> int:
    result = post(
        base_url,
        "/tokenize",
        {
            "model": MODEL,
            "messages": [
                {"role": "user", "content": PREFIX + UNIT * repeats + SUFFIX}
            ],
        },
        300,
    )
    assert result["max_model_len"] == EXPECTED_MAX_MODEL_LEN, result
    return result["count"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:5001")
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    count_1000 = count_tokens(base_url, 1000)
    tokens_per_repeat = count_tokens(base_url, 1001) - count_1000
    assert tokens_per_repeat > 0
    fixed_tokens = count_1000 - 1000 * tokens_per_repeat
    repeats = (TARGET_TOKENS - fixed_tokens) // tokens_per_repeat
    prompt_tokens = count_tokens(base_url, repeats)
    while prompt_tokens > TARGET_TOKENS:
        repeats -= 1
        prompt_tokens = count_tokens(base_url, repeats)
    assert prompt_tokens == EXPECTED_PROMPT_TOKENS, prompt_tokens
    print(f"PROMPT_TOKENS={prompt_tokens}")

    started = time.monotonic()
    result = post(
        base_url,
        "/v1/chat/completions",
        {
            "model": MODEL,
            "messages": [
                {"role": "user", "content": PREFIX + UNIT * repeats + SUFFIX}
            ],
            "temperature": 0,
            "seed": 123,
            "max_tokens": 128,
        },
        1800,
    )
    elapsed = time.monotonic() - started
    answer = result["choices"][0]["message"]["content"]
    assert answer is not None and answer.strip() == NEEDLE, result
    assert result["usage"]["prompt_tokens"] == prompt_tokens, result["usage"]
    print(f"REQUEST_SECONDS={elapsed:.3f}")
    print("PASS long-context retrieval")


if __name__ == "__main__":
    main()
