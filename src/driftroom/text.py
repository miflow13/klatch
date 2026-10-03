"""Token helpers shared by the scheduler and the observer-only analysis."""

import re


def tokens(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.casefold())


def token_set(text: str) -> set[str]:
    return set(tokens(text))


def jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0
