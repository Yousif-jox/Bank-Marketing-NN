# Hotel Booking Cancellation Prediction Using Neural Networks

## AIE231 – Neural Networks Project

**Alamein International University (AIU)**  
**Faculty of Computer Science and Engineering**

---

## 1. Project Overview

This project develops and evaluates neural-network models to predict whether a hotel reservation will be canceled before the guest arrives. Accurate cancellation predictions can help hotels improve inventory planning, reduce revenue loss, and make more informed operational decisions.

The project uses the [Hotel Booking Demand dataset](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand/data) and frames the task as a binary-classification problem.

---

## 2. Problem Statement

Hotel cancellations create uncertainty in room availability, staffing, and revenue management. The goal is to build a model that estimates the likelihood that a booking will be canceled using reservation, guest, and booking-history information available before the booking outcome is known.

The target variable is `is_canceled`:

| Value | Meaning |
|---|---|
| `0` | The booking was not canceled. |
| `1` | The booking was canceled. |

This is a binary-classification task. The model’s predictions can support targeted retention actions and more reliable room-demand planning.

---

## 3. Dataset Description

**Dataset Name:** Hotel Booking Demand

**Source:** Kaggle

**Dataset Link:** https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand/data

The dataset contains hotel-booking records for resort and city hotels. It combines numerical and categorical information about guests, reservation timing, stay characteristics, market segments, and previous cancellations.

### Dataset Information

- Number of records: 119,390
- Target variable: `is_canceled`
- Problem type: Binary classification
- Data types: Numerical and categorical
- Hotel types: Resort hotel and city hotel

### Data Leakage Consideration

To ensure predictions use only information available before the booking outcome, the following columns will be excluded from modeling:

- `reservation_status`
- `reservation_status_date`

Both fields reveal information about the final reservation outcome and would cause target leakage.

### Dataset Challenges

- Class imbalance between canceled and non-canceled bookings.
- Missing values in selected guest and agent/company fields.
- Mixed numerical and categorical features.
- Categorical values with many unique levels.
- Different numerical feature scales.
- Potential leakage from outcome-related columns.

---

## 4. Project Objectives

1. Understand and analyze the Hotel Booking Demand dataset.
2. Perform exploratory data analysis (EDA).
3. Clean and preprocess numerical and categorical features.
4. Exclude leakage-prone reservation-status fields.
5. Build simple baseline classification models.
6. Develop and compare neural-network architectures using TensorFlow/Keras.
7. Experiment with optimizers, regularization, and hyperparameters.
8. Evaluate models with metrics appropriate for cancellation prediction.
9. Analyze incorrect predictions and model limitations.
10. Deliver a reproducible inference pipeline for new booking records.

---

## 5. Technologies and Tools

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Google Colab | Development environment |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Preprocessing and baseline models |
| TensorFlow / Keras | Neural-network development |
| Git & GitHub | Version control and collaboration |

---

## 6. Methodology

### Stage 1: Exploratory Data Analysis

- Inspect dataset structure, data types, and missing values.
- Analyze the distribution of `is_canceled`.
- Examine booking patterns by hotel type, lead time, market segment, deposit type, and customer history.
- Identify outliers and high-cardinality categorical features.

### Stage 2: Data Preprocessing

- Remove `reservation_status` and `reservation_status_date` before modeling.
- Handle missing values and inconsistent categories.
- Encode categorical features and scale numerical features.
- Split data into training, validation, and test sets.
- Build preprocessing steps that are fitted only on the training data.

### Stage 3: Baseline Models

Neural-network models will be compared with simple baselines:

- Majority-class classifier.
- Logistic regression.

### Stage 4: Neural-Network Development

At least three neural-network models will be developed:

1. Shallow multi-layer perceptron (MLP).
2. Deeper MLP with regularization.
3. Best-performing MLP after hyperparameter tuning.

### Stage 5: Model Experimentation

- Compare SGD, SGD with momentum, and Adam optimizers.
- Experiment with learning rates, batch sizes, and architectures.
- Apply dropout, early stopping, and other regularization techniques.
- Conduct ablation studies to measure the contribution of key components.

### Stage 6: Model Evaluation

Evaluate all models on the same held-out test set and analyze common error patterns.

### Stage 7: Inference and Deployment

Create a prediction pipeline that accepts a new booking record and returns a cancellation prediction and probability.

---

## 7. Evaluation Metrics

Since cancellation and non-cancellation classes may be imbalanced, accuracy alone is insufficient.

### Primary Evaluation Metric

**F1-score for the canceled class (`is_canceled = 1`).**

F1-score balances precision and recall, helping assess how effectively the model identifies bookings that are likely to be canceled.

### Additional Evaluation Metrics

- Accuracy
- Precision
- Recall
- Macro F1-score
- ROC-AUC
- Confusion matrix

---

## 8. Week 10: EDA, Preprocessing, and Baselines

Week 10 is complete. The hotel EDA is available in `notebooks/01_eda.ipynb` and includes data-quality checks, class-balance analysis, distributions, IQR outlier flags, correlations, cancellation-rate comparisons, and a feature-availability audit.

### Leakage-Free Preprocessing

`src/data.py` implements the reusable data workflow:

- Separates `is_canceled` from model features.
- Always excludes `reservation_status` and `reservation_status_date`.
- Conservatively excludes `assigned_room_type`, `booking_changes`, `days_in_waiting_list`, `required_car_parking_spaces`, and `total_of_special_requests` for a booking-creation prediction setting.
- Imputes numerical features with the training-set median, then standardizes them.
- Imputes categorical features with `__MISSING__` and one-hot encodes them with unknown-category protection.
- Treats `agent` and `company` as categorical identifiers rather than continuous values.
- Creates stratified 70% / 15% / 15% train, validation, and test splits. The test set is not used for model selection.

### Validation Baselines

Both models were fitted on the training split and evaluated on the same validation split (17,908 rows; 6,633 canceled bookings). F1-score for `is_canceled = 1` is the primary metric.

| Model | Accuracy | Precision | Recall | F1 (canceled) | Macro F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Majority class (`DummyClassifier`) | 0.6296 | 0.0000 | 0.0000 | 0.0000 | 0.3864 | 0.5000 |
| Logistic Regression (`class_weight="balanced"`) | 0.7911 | 0.7030 | 0.7550 | 0.7281 | 0.7792 | 0.8814 |

Logistic Regression substantially improves on the majority baseline for the canceled class. Full classification reports and confusion matrices are saved in `experiments/baseline_report.md`; machine-readable results are in `experiments/baseline_results.csv` and `experiments/baseline_details.json`.

### Reproduction

1. Download the Kaggle CSV to `data/raw/hotel_bookings.csv` as described in `data/README.md`.
2. Install dependencies with `pip install -r requirements.txt`.
3. Run `python -m src.baselines --data data/raw/hotel_bookings.csv --output-dir experiments`.

---

## 9. Team Members and Responsibilities

| No. | Team Member | Role | Main Responsibilities |
|---|---|---|---|
| 1 | Yousif Emad Hassan | Tech Lead + EDA & Baselines | Project coordination, repository management, EDA, and baseline models |
| 2 | Judy Mohammed | Data Engineer | Data cleaning, preprocessing, encoding, and scaling |
| 3 | Ahmed Walid | Model Architect | Neural-network architecture design and implementation |
| 4 | Sara Saeed | Experimentation Engineer | Hyperparameter tuning, optimizer comparison, and ablation studies |
| 5 | Nadine Mohamed | Evaluation & Deployment Engineer | Model evaluation, error analysis, and inference pipeline |

All team members will contribute to neural-network experimentation, documentation, and GitHub collaboration.

---

## 10. Project Structure

```text
Hotel-Booking-Cancellation-NN/
├── README.md
├── requirements.txt
├── app/
├── configs/
├── data/
│   └── README.md
├── experiments/
│   ├── baseline_results.csv
│   ├── baseline_details.json
│   └── baseline_report.md
├── models/
├── notebooks/
│   ├── 01_eda.ipynb
│   └── legacy/
│       ├── README.md
│       └── bank_marketing_eda.ipynb  # archived; not part of the active project
├── report/
└── src/
    ├── data.py
    └── baselines.py
```

The active project uses `notebooks/01_eda.ipynb`. The prior Bank Marketing EDA is retained only as an archived reference under `notebooks/legacy/`.

---

## 11. Expected Outcomes

- Develop multiple neural-network models for hotel-booking cancellation prediction.
- Compare neural networks with traditional baseline models.
- Identify effective preprocessing, architecture, and training choices.
- Measure the impact of regularization and optimization techniques.
- Deliver a reproducible pipeline for predicting booking cancellations.

---

## 12. References

1. [Hotel Booking Demand dataset on Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand/data)
2. Antonio, N., de Almeida, A., & Nunes, L. (2019). Hotel booking demand datasets. *Data in Brief, 22*, 41–49.
3. [TensorFlow Documentation](https://www.tensorflow.org/)
4. [Scikit-learn Documentation](https://scikit-learn.org/)

---

## 13. Project Status

**Current Phase:** Week 10 - Baselines Ready

Completed activities:

- [x] Project topic selected.
- [x] Dataset selected.
- [x] Problem statement defined.
- [x] Team roles assigned.
- [x] GitHub repository created.
- [x] Hotel booking EDA completed.
- [x] Leakage-free preprocessing pipeline implemented.
- [x] Majority-class and Logistic Regression baselines evaluated on validation data.

Upcoming activities:

- [ ] Neural-network development.
- [ ] Model experimentation and evaluation.
- [ ] Final report and presentation.

---

**Course:** AIE231 – Neural Networks  
**University:** Alamein International University (AIU)  
**Project:** Hotel Booking Cancellation Prediction Using Neural Networks
