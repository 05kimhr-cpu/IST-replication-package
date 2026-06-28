"""Metric wrappers.

Iter 1: BLEU-4, ROUGE-L.
Iter 3 adds: METEOR (WordNet-aware), CHRF++ (character-level).

All metrics normalized to [0, 1].
"""
from __future__ import annotations

from dataclasses import dataclass

import sacrebleu
from rouge_score import rouge_scorer

_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

# Lazy-import meteor to avoid nltk download cost when not needed.
_meteor_score = None


def _get_meteor():
    global _meteor_score
    if _meteor_score is None:
        import nltk
        try:
            from nltk.corpus import wordnet  # noqa: F401
            wordnet.ensure_loaded()
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
        from nltk.translate.meteor_score import meteor_score as ms
        _meteor_score = ms
    return _meteor_score


@dataclass(frozen=True)
class Scored:
    metric: str
    score: float


def bleu_sentence(candidate: str, reference: str) -> float:
    b = sacrebleu.sentence_bleu(candidate, [reference])
    return b.score / 100.0


def rougeL(candidate: str, reference: str) -> float:
    return _ROUGE.score(reference, candidate)["rougeL"].fmeasure


def chrf(candidate: str, reference: str) -> float:
    """CHRF++ via sacrebleu defaults (char n-grams 1..6, word n-grams 2).
    Returns [0, 1] (sacrebleu reports 0-100)."""
    return sacrebleu.sentence_chrf(candidate, [reference]).score / 100.0


def meteor(candidate: str, reference: str) -> float:
    """METEOR with WordNet synonym matching. nltk expects tokenized inputs."""
    fn = _get_meteor()
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    return fn([ref_tokens], cand_tokens)


def bertscore_batch(
    candidates: list[str],
    references: list[str],
    model_type: str = "roberta-large",
    device: str = "cuda",
) -> list[float]:
    """Batched BERTScore F1 in [0, 1].

    One warm-up is amortized across the whole batch; call once per experiment.
    """
    import bert_score

    P, R, F1 = bert_score.score(
        candidates,
        references,
        model_type=model_type,
        device=device,
        verbose=False,
        batch_size=64,
        rescale_with_baseline=False,
    )
    return F1.tolist()


def score_all(candidate: str, reference: str) -> list[Scored]:
    return [
        Scored("bleu", bleu_sentence(candidate, reference)),
        Scored("rougeL", rougeL(candidate, reference)),
        Scored("chrf", chrf(candidate, reference)),
        Scored("meteor", meteor(candidate, reference)),
    ]
