import pytest

from amazon_er.metrics.fbeta import entity_fbeta, macro_entity_fbeta


@pytest.mark.parametrize(("truth", "prediction", "expected"), [
    (set(), set(), 1.0),
    (set(), {"A"}, 0.0),
    ({"A"}, set(), 0.0),
    ({"A"}, {"A"}, 1.0),
    ({"A"}, {"A", "B"}, 5 / 9),
    ({"A", "B"}, {"A"}, 5 / 6),
    ({"A", "B"}, {"A", "B"}, 1.0),
])
def test_entity_cases(truth, prediction, expected):
    assert entity_fbeta(truth, prediction) == pytest.approx(expected)


def test_macro_uses_all_s1_including_zero_candidate_entities():
    ground_truth = {"s1": {"A"}, "s2": set(), "s3": {"C"}}
    predictions = {"s1": {"A"}}
    assert macro_entity_fbeta(ground_truth, predictions, ["s1", "s2", "s3"]) == pytest.approx(2 / 3)


def test_universe_is_explicit():
    with pytest.raises(ValueError):
        macro_entity_fbeta({"outside": set()}, {}, ["s1"])
