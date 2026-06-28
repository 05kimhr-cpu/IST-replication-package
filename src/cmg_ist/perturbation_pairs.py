"""Matched synonym/antonym triplet vocabulary for paired perturbation analysis.

Each entry: anchor token -> (synonym substitute, antonym substitute).
Each substitute is a single whitespace-separated word, so every perturbation
is a 1-word substitution at the same position — holding word count constant.
This is the key control: under 1-word substitution at a fixed position, BLEU
and ROUGE-L n-gram breakage is mathematically identical regardless of whether
the substitute preserves or flips meaning. We want to confirm this
empirically.

Coverage expanded beyond iter 1 to raise per-sample paired applicability.
Inflections are spelled out separately so we do not need a morphology layer.
"""
from __future__ import annotations

from dataclasses import dataclass

# (anchor_lower) -> (synonym, antonym)
TRIPLETS: dict[str, tuple[str, str]] = {
    # add / insert / remove family
    "add":     ("insert",    "remove"),
    "adds":    ("inserts",   "removes"),
    "added":   ("inserted",  "removed"),
    "adding":  ("inserting", "removing"),

    "remove":   ("delete",   "insert"),
    "removes":  ("deletes",  "inserts"),
    "removed":  ("deleted",  "inserted"),
    "removing": ("deleting", "inserting"),

    # enable / disable
    "enable":   ("permit",   "disable"),
    "enables":  ("permits",  "disables"),
    "enabled":  ("permitted","disabled"),
    "enabling": ("permitting","disabling"),

    "disable":   ("block",    "enable"),
    "disables":  ("blocks",   "enables"),
    "disabled":  ("blocked",  "enabled"),
    "disabling": ("blocking", "enabling"),

    # show / hide
    "show":    ("display",   "hide"),
    "shows":   ("displays",  "hides"),
    "showed":  ("displayed", "hid"),
    "showing": ("displaying","hiding"),

    "hide":    ("cover",     "show"),
    "hides":   ("covers",    "shows"),
    "hidden":  ("covered",   "shown"),
    "hiding":  ("covering",  "showing"),

    # allow / forbid
    "allow":   ("permit",    "forbid"),
    "allows":  ("permits",   "forbids"),
    "allowed": ("permitted", "forbade"),
    "allowing":("permitting","forbidding"),

    # start / stop
    "start":   ("begin",     "stop"),
    "starts":  ("begins",    "stops"),
    "started": ("began",     "stopped"),
    "starting":("beginning", "stopping"),

    # open / close
    "open":    ("unlock",    "close"),
    "opens":   ("unlocks",   "closes"),
    "opened":  ("unlocked",  "closed"),
    "opening": ("unlocking", "closing"),

    # include / exclude
    "include":   ("contain",   "exclude"),
    "includes":  ("contains",  "excludes"),
    "included":  ("contained", "excluded"),
    "including": ("containing","excluding"),

    # accept / reject
    "accept":   ("take",    "reject"),
    "accepts":  ("takes",   "rejects"),
    "accepted": ("took",    "rejected"),
    "accepting":("taking",  "rejecting"),

    # increase / decrease
    "increase":   ("raise",   "decrease"),
    "increases":  ("raises",  "decreases"),
    "increased":  ("raised",  "decreased"),
    "increasing": ("raising", "decreasing"),

    # create / destroy
    "create":   ("build",   "destroy"),
    "creates":  ("builds",  "destroys"),
    "created":  ("built",   "destroyed"),
    "creating": ("building","destroying"),
}


@dataclass(frozen=True)
class PairApplication:
    anchor: str
    synonym: str
    antonym: str
    anchor_hit_count: int  # how many times anchor appears (case-insensitive)


def apply_pair(msg: str, anchor: str) -> tuple[str, str, PairApplication] | None:
    """Apply both synonym and antonym substitution for the given anchor.

    Returns (syn_candidate, ant_candidate, application_info) or None if the
    anchor is not present in the message.

    All case-insensitive matches of the anchor are replaced, preserving
    leading-capital of the original match. Whitespace structure is preserved.
    """
    if anchor not in TRIPLETS:
        raise KeyError(anchor)
    syn, ant = TRIPLETS[anchor]

    # Find all case-insensitive whole-word occurrences.
    import re

    pattern = re.compile(rf"(?<![A-Za-z]){re.escape(anchor)}(?![A-Za-z])", re.IGNORECASE)
    matches = pattern.findall(msg)
    if not matches:
        return None

    def _replace(m: "re.Match[str]") -> str:
        matched = m.group(0)
        # placeholder — actual choice of syn/ant handled by two separate subs
        return matched

    def _make(sub: str) -> str:
        def _repl(m: "re.Match[str]") -> str:
            matched = m.group(0)
            if matched[:1].isupper():
                return sub[:1].upper() + sub[1:]
            return sub

        return pattern.sub(_repl, msg)

    syn_out = _make(syn)
    ant_out = _make(ant)
    return syn_out, ant_out, PairApplication(
        anchor=anchor,
        synonym=syn,
        antonym=ant,
        anchor_hit_count=len(matches),
    )
