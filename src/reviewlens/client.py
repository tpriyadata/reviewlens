"""Client factory. Reads config from the environment (.env locally)."""
from __future__ import annotations

import os

from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv


class ConfigError(RuntimeError):
    pass


def build_client() -> TextAnalyticsClient:
    # utf-8-sig tolerates a BOM in .env (PowerShell 5.1 Set-Content adds one).
    load_dotenv(encoding="utf-8-sig")
    endpoint = os.getenv("LANGUAGE_ENDPOINT", "").strip()
    key = os.getenv("LANGUAGE_KEY", "").strip()
    if not endpoint or not key:
        raise ConfigError("LANGUAGE_ENDPOINT and LANGUAGE_KEY must be set (see .env.example).")
    if not endpoint.startswith("https://"):
        raise ConfigError("LANGUAGE_ENDPOINT must start with https://")
    # azure-core's RetryPolicy already retries 429/5xx and honours Retry-After.
    return TextAnalyticsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key),
        retry_total=5,
        retry_backoff_factor=1.0,
    )
