"""Smoke test: confirms credentials work and sentiment + opinion mining return results."""
import os
import sys

from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
    ServiceRequestError,
)
from dotenv import load_dotenv

load_dotenv()

endpoint = os.getenv("LANGUAGE_ENDPOINT")
key = os.getenv("LANGUAGE_KEY")
if not endpoint or not key:
    sys.exit("Missing LANGUAGE_ENDPOINT or LANGUAGE_KEY in .env")

client = TextAnalyticsClient(endpoint=endpoint, credential=AzureKeyCredential(key))

reviews = [
    "The battery dies fast but the screen is gorgeous.",
    "Delivery was quick and the packaging was great.",
    "Customer support never replied. Terrible experience.",
]

try:
    results = client.analyze_sentiment(reviews, show_opinion_mining=True)
except ClientAuthenticationError:
    sys.exit("Auth failed: check LANGUAGE_KEY matches this endpoint's resource.")
except ServiceRequestError:
    sys.exit("Can't reach endpoint: check LANGUAGE_ENDPOINT URL.")
except HttpResponseError as e:
    sys.exit(f"Service error {e.status_code}: {e.message}")

for review, doc in zip(reviews, results):
    print(f"\n{review}")
    if doc.is_error:
        print(f"  ERROR: {doc.error.code} - {doc.error.message}")
        continue
    s = doc.confidence_scores
    print(f"  Overall: {doc.sentiment}  (pos={s.positive:.2f} neu={s.neutral:.2f} neg={s.negative:.2f})")
    for sentence in doc.sentences:
        for op in sentence.mined_opinions:
            words = ", ".join(a.text for a in op.assessments)
            print(f"  Aspect: {op.target.text} -> {op.target.sentiment}  [{words}]")