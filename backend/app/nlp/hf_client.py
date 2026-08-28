"""
hf_client.py

Small shared helper for calling the Hugging Face Inference API instead
of loading models locally.

Why this exists: running bart-large-mnli / distilbart locally on a
512MB-RAM free-tier host was crashing the process with an OOM as soon
as /analyze was hit (Render's own logs confirmed the process getting
killed mid-request). Offloading the actual model inference to HF's
hosted Inference API means our server process only ever needs
`requests` in memory for these calls - the heavy torch/transformers
forward passes happen on HF's infrastructure, not ours.

Requires HF_API_TOKEN to be set in the environment (a free "Read"
token from huggingface.co/settings/tokens is enough).
"""

import os
import time
import requests

HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_API_URL = "https://api-inference.huggingface.co/models/{model}"

# serverless HF models "cold start" the first time they're hit after a
# period of inactivity - the API returns a 503 with an estimated_time
# while it's loading the model on their end. We poll/retry instead of
# failing immediately, since this is a one-time cost per model per
# period of inactivity, not something wrong with our request.
MAX_RETRIES = 6
RETRY_WAIT_SECONDS = 10


def query(model: str, payload: dict) -> dict:
    if not HF_API_TOKEN:
        raise RuntimeError(
            "HF_API_TOKEN is not set. Add it as an environment variable "
            "(a free token from huggingface.co/settings/tokens with "
            "'Read' access is enough)."
        )

    url = HF_API_URL.format(model=model)
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

    last_error_body = None

    for attempt in range(MAX_RETRIES):
        response = requests.post(url, headers=headers, json=payload, timeout=60)

        if response.status_code == 200:
            return response.json()

        if response.status_code == 503:
            # model is loading on HF's side - wait and retry
            try:
                wait_hint = response.json().get("estimated_time", RETRY_WAIT_SECONDS)
            except Exception:
                wait_hint = RETRY_WAIT_SECONDS
            time.sleep(min(wait_hint, 20))
            continue

        # anything else (400/401/429/etc) - no point retrying blindly
        last_error_body = response.text
        break

    raise RuntimeError(
        f"Hugging Face Inference API call failed for model '{model}': "
        f"{last_error_body or 'model did not finish loading in time'}"
    )
