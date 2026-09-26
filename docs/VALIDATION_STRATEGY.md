# Validation Strategy

The primary split is a grouped S1 holdout so records associated with a reference entity do not leak across folds. Cross-country robustness will also use leave-one-country-out checks: train US/evaluate India and train India/evaluate US.

Where computationally feasible, evaluation must include the full realistic target distractor universe. Candidate evaluation reports pair recall, anchor hit rate, complete-set coverage, candidate-oracle macro-F0.5, mean candidate count, and p95 candidate count.

Final evaluation reports entity macro-F0.5, precision, recall, singleton accuracy, source-specific F0.5, and country-specific F0.5. It begins with every S1 reference ID and left-joins predictions; an S1 with no accepted candidate receives an empty predicted set.

Pair classification performance is diagnostic, not the final objective.
