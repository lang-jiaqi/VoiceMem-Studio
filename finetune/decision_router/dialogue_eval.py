"""Semantic slices and group metrics for independently held-out dialogue rows."""
from __future__ import annotations

from collections import defaultdict
import statistics


def summarize(rows, predicted):
    """Report both label errors and semantic slices, never only aggregate accuracy."""
    if len(rows) != len(predicted) or not rows:
        raise ValueError('Expected a prediction for every nonempty evaluation row')
    correct = [bool(value) == row['needs_deep'] for row, value in zip(rows, predicted)]
    groups, categories = defaultdict(list), defaultdict(list)
    for index, row in enumerate(rows):
        groups[row['group']].append(correct[index])
        categories[row.get('category', 'legacy')].append(index)

    def measure(indices):
        selected = [rows[index] for index in indices]
        guesses = [bool(predicted[index]) for index in indices]
        positives = [value for row, value in zip(selected, guesses) if row['needs_deep']]
        negatives = [value for row, value in zip(selected, guesses) if not row['needs_deep']]
        return {'n': len(indices), 'correct': sum(correct[index] for index in indices),
                'accuracy': statistics.mean(correct[index] for index in indices),
                'deep_recall': statistics.mean(positives) if positives else None,
                'false_deep_rate': statistics.mean(negatives) if negatives else None}

    result = measure(list(range(len(rows))))
    result.update(groups=len(groups),
                  macro_group_accuracy=statistics.mean(statistics.mean(values) for values in groups.values()),
                  by_category={name: measure(indices) for name, indices in sorted(categories.items())})
    for name, predicate in (('with_history', lambda row: bool(row.get('history'))),
                            ('without_history', lambda row: not row.get('history')),
                            ('observed_audio', lambda row: bool(row.get('audio_state')))):
        indices = [index for index, row in enumerate(rows) if predicate(row)]
        if indices:
            result[name] = measure(indices)
    return result


def record_predictions(records):
    """Upstream records preserve canonical ordinary/deep order and row order."""
    return [int(record[1].argmax()) == 1 for record in records]


def validation_score(measured):
    """Give every semantic family equal influence on checkpoint selection."""
    categories = measured['by_category']
    accuracy = statistics.mean(value['accuracy'] for value in categories.values())
    return accuracy, measured['macro_group_accuracy'], measured['accuracy']
