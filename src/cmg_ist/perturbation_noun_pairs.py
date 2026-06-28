"""Iter 7 — target-noun perturbation vocabulary.

Design intent: test whether the meaning-direction blindness observed for
verb substitutions in iter 2–5 generalizes to *noun* substitutions. Nouns
don't have crisp antonyms the way action verbs do; instead we contrast a
**near-synonym** (preserves the referent of the commit's target object)
against a **reference-disjoint** noun (a different software concept, so
the commit's claim is now about a different thing).

Triplet format: anchor -> (near_synonym, reference_disjoint).

Caveats:
- Noun synonymy/antonymy in software jargon is fuzzier than verb pairs.
- "Meaning-change" here is claim-referent change, not polar flip.
- We tag each triplet with a kind so per-kind analysis can separate cleaner
  pairs (e.g., import/export is a directional opposite) from looser pairs
  (e.g., function/variable is "different kind of program entity").
"""
from __future__ import annotations

from dataclasses import dataclass

# anchor_lower -> (near_synonym, reference_disjoint, kind)
_RAW: dict[str, tuple[str, str, str]] = {
    # close program entities
    "function":  ("method",    "variable",  "close_entity"),
    "functions": ("methods",   "variables", "close_entity"),
    "method":    ("function",  "attribute", "close_entity"),
    "methods":   ("functions", "attributes","close_entity"),
    "class":     ("type",      "module",    "close_entity"),
    "classes":   ("types",     "modules",   "close_entity"),
    "parameter": ("argument",  "constant",  "close_entity"),
    "parameters":("arguments", "constants", "close_entity"),
    "argument":  ("parameter", "keyword",   "close_entity"),
    "arguments": ("parameters","keywords",  "close_entity"),

    # diagnostics
    "error":     ("exception", "warning",   "diagnostic"),
    "errors":    ("exceptions","warnings",  "diagnostic"),
    "bug":       ("issue",     "feature",   "diagnostic"),
    "bugs":      ("issues",    "features",  "diagnostic"),
    "warning":   ("alert",     "error",     "diagnostic"),
    "warnings":  ("alerts",    "errors",    "diagnostic"),

    # infrastructure
    "test":      ("check",     "config",    "infrastructure"),
    "tests":     ("checks",    "configs",   "infrastructure"),
    "log":       ("trace",     "output",    "infrastructure"),
    "logs":      ("traces",    "outputs",   "infrastructure"),
    "comment":   ("note",      "command",   "infrastructure"),
    "comments":  ("notes",     "commands",  "infrastructure"),

    # directional (cleanest "flip")
    "import":    ("include",   "export",    "directional"),
    "imports":   ("includes",  "exports",   "directional"),
    "input":     ("entry",     "output",    "directional"),
    "inputs":    ("entries",   "outputs",   "directional"),
}


TRIPLETS_NOUN: dict[str, tuple[str, str]] = {k: (v[0], v[1]) for k, v in _RAW.items()}
KINDS: dict[str, str] = {k: v[2] for k, v in _RAW.items()}


@dataclass(frozen=True)
class NounApplication:
    anchor: str
    synonym: str
    disjoint: str
    kind: str
    anchor_hit_count: int


def apply_noun_pair(msg: str, anchor: str) -> tuple[str, str, NounApplication] | None:
    """Substitute anchor with synonym and with disjoint referent.

    Returns (syn_candidate, disjoint_candidate, info) or None if anchor is
    not present as a whole word.
    """
    if anchor not in TRIPLETS_NOUN:
        raise KeyError(anchor)
    syn, dis = TRIPLETS_NOUN[anchor]
    kind = KINDS[anchor]

    import re
    pattern = re.compile(rf"(?<![A-Za-z]){re.escape(anchor)}(?![A-Za-z])", re.IGNORECASE)
    matches = pattern.findall(msg)
    if not matches:
        return None

    def _make(sub: str) -> str:
        def _repl(m: "re.Match[str]") -> str:
            matched = m.group(0)
            if matched[:1].isupper():
                return sub[:1].upper() + sub[1:]
            return sub
        return pattern.sub(_repl, msg)

    return _make(syn), _make(dis), NounApplication(
        anchor=anchor, synonym=syn, disjoint=dis,
        kind=kind, anchor_hit_count=len(matches),
    )
