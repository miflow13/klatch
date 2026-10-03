"""Shared token-set helpers used by the scheduler and the observer-only analysis."""

import pytest

from driftroom.text import jaccard, token_set


def test_token_set_is_casefolded_word_tokens() -> None:
    assert token_set("The HARBOR, the harbor! it's") == {"the", "harbor", "it", "s"}
    assert token_set("") == set()


def test_jaccard_is_intersection_over_union_and_zero_for_two_empty_sets() -> None:
    assert jaccard({"a", "b", "c"}, {"b", "c", "d"}) == pytest.approx(0.5)
    assert jaccard({"a"}, {"a"}) == 1.0
    assert jaccard({"a"}, {"b"}) == 0.0
    assert jaccard(set(), set()) == 0.0
    assert jaccard(set(), {"a"}) == 0.0
