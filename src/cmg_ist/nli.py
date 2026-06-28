"""Minimal NLI wrapper for the iter 5b discriminative probe.

Uses facebook/bart-large-mnli (contradiction=0, neutral=1, entailment=2).
Batched inference on CUDA. Returns entail / contradict / neutral probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = "facebook/bart-large-mnli"

_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL).to("cuda").eval()
    return _tokenizer, _model


@dataclass(frozen=True)
class NLIResult:
    contradiction: float
    neutral: float
    entailment: float


def batch_nli(
    premises: list[str],
    hypotheses: list[str],
    batch_size: int = 64,
) -> list[NLIResult]:
    tok, mdl = _load()
    results: list[NLIResult] = []
    n = len(premises)
    assert len(hypotheses) == n
    for i in range(0, n, batch_size):
        prem = premises[i : i + batch_size]
        hyp = hypotheses[i : i + batch_size]
        inp = tok(prem, hyp, return_tensors="pt", truncation=True, padding=True).to("cuda")
        with torch.no_grad():
            logits = mdl(**inp).logits
        probs = torch.softmax(logits, dim=-1).cpu().tolist()
        for p in probs:
            results.append(NLIResult(contradiction=p[0], neutral=p[1], entailment=p[2]))
    return results
