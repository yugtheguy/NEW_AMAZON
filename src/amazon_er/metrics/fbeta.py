"""Exact project-contract entity set F-beta metric."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping


def entity_fbeta(ground_truth: Iterable[Hashable], prediction: Iterable[Hashable], beta: float = 0.5) -> float:
    if beta <= 0:
        raise ValueError("beta must be positive")
    truth, predicted = set(ground_truth), set(prediction)
    if not truth:
        return 1.0 if not predicted else 0.0
    true_positive = len(truth & predicted)
    if true_positive == 0:
        return 0.0
    precision = true_positive / len(predicted)
    recall = true_positive / len(truth)
    beta_sq = beta * beta
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def macro_entity_fbeta(
    ground_truth: Mapping[Hashable, Iterable[Hashable]],
    predictions: Mapping[Hashable, Iterable[Hashable]],
    evaluation_universe: Iterable[Hashable],
    beta: float = 0.5,
) -> float:
    entities = list(evaluation_universe)
    if not entities:
        raise ValueError("evaluation_universe must contain at least one S1 entity")
    if len(set(entities)) != len(entities):
        raise ValueError("evaluation_universe contains duplicate S1 entities")
    unknown_gt = set(ground_truth) - set(entities)
    unknown_pred = set(predictions) - set(entities)
    if unknown_gt or unknown_pred:
        raise ValueError(f"Mappings contain entities outside evaluation universe: gt={unknown_gt}, pred={unknown_pred}")
    scores = [entity_fbeta(ground_truth.get(entity, ()), predictions.get(entity, ()), beta) for entity in entities]
    return sum(scores) / len(scores)
