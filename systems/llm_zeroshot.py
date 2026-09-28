"""
System 3: LLM zero-shot.

Send each utterance to Claude with the list of 77 intents and ask for exactly
one label. No training data, no examples.

Three decisions fixed in advance, before any result on EVAL is seen:

1. OUTPUT VALIDATION IS STRICT. The response is normalised (whitespace,
   surrounding quotes/backticks/periods, case) and must then match one of the 77
   labels exactly. Anything else is recorded as INVALID and scored as wrong.
   No fuzzy matching, no "closest label" rescue — that would be tuning the
   scorer to the model. The invalid rate is reported as its own metric.

2. Case-insensitive matching maps back to the canonical label. One Banking77
   label ("Refund_not_showing_up") is capitalised; without this, a correct
   classification would score as wrong for a formatting reason.

3. Temperature 0, fixed prompt version, fixed model string. Prompt iteration
   happens on DEV only. Every version tried is logged in PROMPT_VERSIONS.

Responses are cached to disk keyed by (model, prompt_version, example_id), so a
crashed run resumes without re-billing and a rerun is reproducible.

API key is read from the ANTHROPIC_API_KEY environment variable. Never commit it.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .base import Prediction, System

MODEL = "claude-haiku-4-5"
INVALID = "__INVALID__"
CACHE_DIR = Path("data/llm_cache")

# Log every prompt variant tried on DEV. This is the honest denominator for
# dev overfitting: the more variants, the more the chosen one is fitted to noise.
PROMPT_VERSIONS = {
    "v1": "baseline: task statement + label list, answer with label only",
}
PROMPT_VERSION = "v1"


def build_system_prompt(labels: list[str]) -> str:
    label_block = "\n".join(labels)
    return (
        "You classify customer messages sent to a bank's support channel.\n\n"
        "Assign each message to exactly one of the following intents:\n\n"
        f"{label_block}\n\n"
        "Respond with the intent name only, copied exactly as written above. "
        "No explanation, no punctuation, no other text."
    )


def normalise(text: str) -> str:
    return text.strip().strip("`'\".").strip().lower()


class LLMZeroShot(System):
    name = "llm_zeroshot"

    def __init__(self, labels_json="data/processed/labels.json",
                 model: str = MODEL, max_workers: int = 8, max_retries: int = 5):
        import anthropic  # imported here so other systems don't need the SDK
        self.client = anthropic.Anthropic()
        self.labels = json.loads(Path(labels_json).read_text())
        self.lookup = {l.lower(): l for l in self.labels}
        self.model = model
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.system_prompt = build_system_prompt(self.labels)
        self.cache = CACHE_DIR / f"{model}__{PROMPT_VERSION}.jsonl"
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        self._cached = self._load_cache()

    def config(self) -> dict:
        return {
            "model": self.model,
            "prompt_version": PROMPT_VERSION,
            "prompt_versions_tried": list(PROMPT_VERSIONS),
            "temperature": 0,
            "validation": "strict: normalise then exact match, else INVALID",
            "system_prompt_sha1": hashlib.sha1(self.system_prompt.encode()).hexdigest()[:12],
        }

    # -- cache ---------------------------------------------------------------
    def _load_cache(self) -> dict:
        if not self.cache.exists():
            return {}
        out = {}
        for line in self.cache.read_text().splitlines():
            rec = json.loads(line)
            out[rec["example_id"]] = rec
        return out

    def _append_cache(self, rec: dict) -> None:
        with self.cache.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    # -- one call ------------------------------------------------------------
    def _call(self, text: str) -> dict:
        delay = 1.0
        for attempt in range(self.max_retries):
            try:
                t0 = time.perf_counter()
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=32,
                    temperature=0,
                    system=self.system_prompt,
                    messages=[{"role": "user", "content": text}],
                )
                ms = (time.perf_counter() - t0) * 1000
                raw = "".join(b.text for b in resp.content if b.type == "text")
                return {
                    "raw_output": raw,
                    "latency_ms": ms,
                    "input_tokens": resp.usage.input_tokens,
                    "output_tokens": resp.usage.output_tokens,
                }
            except Exception as e:                      # rate limit, overload, network
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(delay)
                delay *= 2

    def _predict_one(self, eid: str, text: str) -> Prediction:
        if eid in self._cached:
            rec = self._cached[eid]
        else:
            rec = {"example_id": eid, **self._call(text)}
            self._append_cache(rec)
        label = self.lookup.get(normalise(rec["raw_output"]), INVALID)
        return Prediction(eid, label, rec["latency_ms"],
                          rec["input_tokens"], rec["output_tokens"], rec["raw_output"])

    def predict(self, ids, texts) -> list[Prediction]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            preds = list(pool.map(self._predict_one, ids, texts))
        n_invalid = sum(p.predicted_label == INVALID for p in preds)
        print(f"  invalid outputs: {n_invalid}/{len(preds)} "
              f"({n_invalid / len(preds):.2%})")
        return preds
