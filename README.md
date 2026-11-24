# Probability Calibration Lab

Measure how closely binary probabilities match observed outcomes. The lab searches for a single temperature on a validation split, divides logits by that temperature, and evaluates the adjusted probabilities on a separate test split. Positive temperatures preserve the decision boundary.

## Run

Use Python 3.10 or newer. Run from this repository with the standard library.

```sh
python calibration.py
python calibration.py --validation validation.csv --test test.csv --bins 10
python -m unittest -v
```

Each CSV needs a finite `logit` and a binary `label` column. Prepare disjoint files using predictions from the same fitted model.

## Demo result

The synthetic example inflates scores to simulate an overconfident model. The validation search selects a temperature of **2.45** using 1,600 examples. A separate set of 1,600 gives the following results.

| Metric | Raw | Calibrated |
| --- | ---: | ---: |
| Accuracy | 71.94% | 71.94% |
| Binary log loss | 0.6858 | 0.5398 |
| Brier score | 0.2065 | 0.1826 |
| Expected calibration error | 0.1407 | 0.0311 |

The JSON report includes the mean predicted probability and observed positive rate for each occupied bin. Calibration error depends on how the bins are defined, and a single temperature can miss different error patterns across groups. An extension could report calibration by segment on real held-out predictions.

This is an AI-assisted portfolio project using synthetic examples. [Guo et al., On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html)
