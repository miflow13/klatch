"""Shared token-set helpers used by the scheduler and the observer-only analysis."""

import pytest

from driftroom.text import echoes_recent, jaccard, token_set


def test_token_set_is_casefolded_word_tokens() -> None:
    assert token_set("The HARBOR, the harbor! it's") == {"the", "harbor", "it", "s"}
    assert token_set("") == set()


def test_jaccard_is_intersection_over_union_and_zero_for_two_empty_sets() -> None:
    assert jaccard({"a", "b", "c"}, {"b", "c", "d"}) == pytest.approx(0.5)
    assert jaccard({"a"}, {"a"}) == 1.0
    assert jaccard({"a"}, {"b"}) == 0.0
    assert jaccard(set(), set()) == 0.0
    assert jaccard(set(), {"a"}) == 0.0


def test_echoes_recent_checks_only_the_window_before_the_index() -> None:
    sets = [{"a", "b"}, {"x", "y"}, {"p", "q"}, {"a", "b"}]

    assert echoes_recent(3, sets, 3, 0.6)  # index 0 is the oldest of the 3 before it
    assert not echoes_recent(3, sets, 2, 0.6)  # index 0 is outside a window of 2
    assert not echoes_recent(1, sets, 5, 0.6)  # dissimilar to everything before it
    assert not echoes_recent(0, sets, 5, 0.6)  # nothing before it
    assert echoes_recent(3, sets, 5, 1.0)  # an identical set meets a threshold of 1


def test_echoes_recent_ignores_later_items_and_uses_a_greater_or_equal_threshold() -> None:
    sets = [{"a", "b", "c"}, {"a", "b", "c", "d", "e"}, {"a", "b", "c"}]

    assert echoes_recent(1, sets, 1, 0.6)  # Jaccard exactly 0.6
    assert not echoes_recent(1, sets, 1, 0.61)
