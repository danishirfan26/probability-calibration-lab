"""Fit a temperature on validation logits and report held-out calibration."""

import argparse
import csv
import json
import math
import random
from pathlib import Path
from statistics import fmean


def sigmoid(value):
    if value < 0:
        exponential = math.exp(value)
        return exponential / (1 + exponential)
    return 1 / (1 + math.exp(-value))


def loss(logit, label):
    """Compute binary log loss from a logit without clipping probabilities."""
    signed_margin = logit if label else -logit
    return max(0, -signed_margin) + math.log1p(math.exp(-abs(signed_margin)))


def fit_temperature(validation):
    if not validation:
        raise ValueError("Validation data must be nonempty.")
    candidates = [step / 20 for step in range(5, 101)]
    return min(candidates, key=lambda temp: fmean(loss(logit / temp, y) for logit, y in validation))


def evaluate(rows, temperature=1.0, bins=10):
    if not rows or not math.isfinite(temperature) or temperature <= 0 or not 2 <= bins <= 100:
        raise ValueError("Use nonempty test data, a positive finite temperature, and 2–100 bins.")
    groups = [[] for _ in range(bins)]
    probabilities, losses = [], []
    correct = 0
    for logit, label in rows:
        adjusted = logit / temperature
        probability = sigmoid(adjusted)
        groups[min(int(probability * bins), bins - 1)].append((probability, label))
        probabilities.append((probability, label))
        losses.append(loss(adjusted, label))
        correct += (adjusted >= 0) == label

    reliability = []
    ece = 0.0
    for index, group in enumerate(groups):
        if not group:
            continue
        mean_probability = fmean(probability for probability, _ in group)
        positive_rate = fmean(label for _, label in group)
        ece += len(group) / len(rows) * abs(mean_probability - positive_rate)
        reliability.append({
            "lower": round(index / bins, 4),
            "upper": round((index + 1) / bins, 4),
            "count": len(group),
            "mean_probability": round(mean_probability, 4),
            "observed_positive_rate": round(positive_rate, 4),
        })
    return {
        "accuracy": round(correct / len(rows), 4),
        "log_loss": round(fmean(losses), 4),
        "brier_score": round(fmean((probability - label) ** 2 for probability, label in probabilities), 4),
        "ece": round(ece, 4),
        "reliability": reliability,
    }


def synthetic_batches(seed=58):
    rng = random.Random(seed)
    examples = []
    for _ in range(3200):
        true_logit = rng.gauss(0, 1.4)
        label = int(rng.random() < sigmoid(true_logit))
        examples.append((2.6 * true_logit, label))
    return examples[:1600], examples[1600:]


def read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, strict=True)
        if not reader.fieldnames or not {"logit", "label"} <= set(reader.fieldnames):
            raise ValueError("CSV requires logit and label columns.")
        result = []
        for row in reader:
            try:
                logit = float(row["logit"])
                label = int(row["label"])
                if not math.isfinite(logit) or label not in (0, 1):
                    raise ValueError()
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid logit or binary label on CSV line {reader.line_num}.") from exc
            result.append((logit, label))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=58)
    args = parser.parse_args()
    try:
        if args.validation and args.test:
            if args.validation.resolve() == args.test.resolve():
                raise ValueError("Validation and test must use distinct files.")
            validation, test = read_rows(args.validation), read_rows(args.test)
        elif args.validation or args.test:
            raise ValueError("Provide both validation and test CSV files.")
        else:
            validation, test = synthetic_batches(args.seed)
        temperature = fit_temperature(validation)
        report = {
            "validation_size": len(validation),
            "test_size": len(test),
            "temperature": temperature,
            "before": evaluate(test, bins=args.bins),
            "after": evaluate(test, temperature, args.bins),
        }
    except (OSError, csv.Error, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
