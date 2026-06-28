"""Candidate-message perturbations for iteration 1.

Design intent (see docs/iteration_01_plan.md):
  - *Trivial* perturbations should not change what a human considers the
    message, and therefore should not move a well-designed metric.
  - *Paraphrase* perturbations preserve meaning but change surface form,
    and should move surface-overlap metrics while remaining above a
    meaning-change floor.
  - *Meaning-change* perturbations alter the claim; a useful metric should
    drop sharply.

All perturbations are deterministic given the input string, so the same
sample always yields the same perturbed output. No randomness in iter 1.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

# --- trivial -----------------------------------------------------------------

def pert_whitespace_double(msg: str) -> str:
    """Replace single spaces with double spaces. Human-trivial."""
    return re.sub(r" ", "  ", msg)


def pert_trailing_period_flip(msg: str) -> str:
    """Toggle trailing period."""
    s = msg.rstrip()
    if s.endswith("."):
        return s[:-1]
    return s + "."


# --- paraphrase (meaning-preserving lexical swap) ----------------------------
# Conservative swaps where the near-synonym is safe for commit-msg context.
PARAPHRASE_PAIRS: dict[str, str] = {
    # additions
    "add": "introduce",
    "added": "introduced",
    "adds": "introduces",
    "adding": "introducing",
    # removals
    "remove": "delete",
    "removed": "deleted",
    "removes": "deletes",
    "removing": "deleting",
    # fixes
    "fix": "correct",
    "fixed": "corrected",
    "fixes": "corrects",
    "fixing": "correcting",
    # updates
    "update": "modify",
    "updated": "modified",
    "updates": "modifies",
    "updating": "modifying",
}


def _tokenwise_replace(msg: str, mapping: dict[str, str]) -> str:
    tokens = re.split(r"(\s+)", msg)  # keep spaces
    out = []
    for tok in tokens:
        low = tok.lower()
        if low in mapping:
            repl = mapping[low]
            # preserve simple capitalization
            if tok[:1].isupper():
                repl = repl[:1].upper() + repl[1:]
            out.append(repl)
        else:
            out.append(tok)
    return "".join(out)


def pert_paraphrase_synonym(msg: str) -> str:
    """Meaning-preserving synonym swap on first-class action verbs."""
    return _tokenwise_replace(msg, PARAPHRASE_PAIRS)


# --- meaning-changing --------------------------------------------------------
# Antonyms: swap add<->remove, enable<->disable. A message that used to say
# "add X" now says "remove X" — a different claim about the diff.
ANTONYM_PAIRS: dict[str, str] = {
    "add": "remove",
    "added": "removed",
    "adds": "removes",
    "adding": "removing",
    "remove": "add",
    "removed": "added",
    "removes": "adds",
    "removing": "adding",
    "enable": "disable",
    "enabled": "disabled",
    "enables": "disables",
    "enabling": "disabling",
    "disable": "enable",
    "disabled": "enabled",
    "disables": "enables",
    "disabling": "enabling",
}


def pert_action_antonym(msg: str) -> str:
    """Meaning-changing antonym swap on action verbs. Only applies if present."""
    return _tokenwise_replace(msg, ANTONYM_PAIRS)


# --- registry ----------------------------------------------------------------

@dataclass(frozen=True)
class Perturbation:
    name: str
    kind: str  # "trivial" | "paraphrase" | "meaning_change"
    fn: Callable[[str], str]


REGISTRY: list[Perturbation] = [
    Perturbation("whitespace_double", "trivial", pert_whitespace_double),
    Perturbation("trailing_period_flip", "trivial", pert_trailing_period_flip),
    Perturbation("synonym_swap", "paraphrase", pert_paraphrase_synonym),
    Perturbation("action_antonym", "meaning_change", pert_action_antonym),
]


def apply(msg: str, name: str) -> tuple[str, bool]:
    """Apply a named perturbation. Returns (result, applied_meaningfully).

    `applied_meaningfully` is False if the output equals the input — e.g. the
    synonym swap found no target token. Downstream analysis should filter
    non-applicable rows out so we don't confound the averages with no-ops.
    """
    pert = next((p for p in REGISTRY if p.name == name), None)
    if pert is None:
        raise KeyError(name)
    out = pert.fn(msg)
    return out, out != msg
