"""Token helpers shared by the scheduler and the observer-only analysis."""

from collections.abc import Sequence
import re


def tokens(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.casefold())


def token_set(text: str) -> set[str]:
    return set(tokens(text))


def jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def echoes_recent(
    index: int, token_sets: Sequence[set[str]], window: int, threshold: float
) -> bool:
    """True when ``token_sets[index]`` is at least ``threshold`` alike (Jaccard) to any
    of the ``window`` token sets just before it."""
    return any(
        jaccard(token_sets[index], earlier) >= threshold
        for earlier in token_sets[max(0, index - window):index]
    )
