# Week 10 Baseline Validation Results

These metrics were calculated on the validation split only. The test split was not used for model selection.

## Split distribution

| split | rows | canceled_count | canceled_rate |
|---|---|---|---|
| train | 83573 | 30957 | 0.3704 |
| validation | 17908 | 6633 | 0.3704 |
| test | 17909 | 6634 | 0.3704 |

## Model comparison

| model | accuracy | precision_canceled | recall_canceled | f1_canceled | macro_f1 | roc_auc |
|---|---|---|---|---|---|---|
| logistic_regression_balanced | 0.7911 | 0.7030 | 0.7550 | 0.7281 | 0.7792 | 0.8814 |
| dummy_most_frequent | 0.6296 | 0.0000 | 0.0000 | 0.0000 | 0.3864 | 0.5000 |

## Confusion matrices

### dummy_most_frequent

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 11275 | 0 |
| Actual 1 | 6633 | 0 |

Classification report:

```json
{
  "0": {
    "precision": 0.6296068796068796,
    "recall": 1.0,
    "f1-score": 0.7727101394647569,
    "support": 11275.0
  },
  "1": {
    "precision": 0.0,
    "recall": 0.0,
    "f1-score": 0.0,
    "support": 6633.0
  },
  "accuracy": 0.6296068796068796,
  "macro avg": {
    "precision": 0.3148034398034398,
    "recall": 0.5,
    "f1-score": 0.38635506973237843,
    "support": 17908.0
  },
  "weighted avg": {
    "precision": 0.39640482284831174,
    "recall": 0.6296068796068796,
    "f1-score": 0.48650361974900236,
    "support": 17908.0
  }
}
```

### logistic_regression_balanced

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 9159 | 2116 |
| Actual 1 | 1625 | 5008 |

Classification report:

```json
{
  "0": {
    "precision": 0.8493137982195845,
    "recall": 0.8123281596452329,
    "f1-score": 0.8304093567251462,
    "support": 11275.0
  },
  "1": {
    "precision": 0.7029758562605278,
    "recall": 0.7550128147143073,
    "f1-score": 0.728065712001163,
    "support": 6633.0
  },
  "accuracy": 0.7910989501898593,
  "macro avg": {
    "precision": 0.7761448272400562,
    "recall": 0.7836704871797701,
    "f1-score": 0.7792375343631546,
    "support": 17908.0
  },
  "weighted avg": {
    "precision": 0.7951112312654621,
    "recall": 0.7910989501898593,
    "f1-score": 0.7925019748034252,
    "support": 17908.0
  }
}
```
