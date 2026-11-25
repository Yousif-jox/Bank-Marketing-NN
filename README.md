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

## 8. Team Members and Responsibilities

| No. | Team Member | Role | Main Responsibilities |
|---|---|---|---|
| 1 | Yousif Emad Hassan | Tech Lead + EDA & Baselines | Project coordination, repository management, EDA, and baseline models |
| 2 | Judy Mohammed | Data Engineer | Data cleaning, preprocessing, encoding, and scaling |
| 3 | Ahmed Walid | Model Architect | Neural-network architecture design and implementation |
| 4 | Sara Saeed | Experimentation Engineer | Hyperparameter tuning, optimizer comparison, and ablation studies |
| 5 | Nadine Mohamed | Evaluation & Deployment Engineer | Model evaluation, error analysis, and inference pipeline |

All team members will contribute to neural-network experimentation, documentation, and GitHub collaboration.

---

## 9. Project Structure

```text
Bank-Marketing-NN/
├── README.md
├── requirements.txt
├── app/
├── configs/
├── data/
│   └── README.md
├── experiments/
├── models/
├── notebooks/
│   └── 01_eda.ipynb.ipynb
├── report/
└── src/
```

---

## 10. Expected Outcomes

- Develop multiple neural-network models for hotel-booking cancellation prediction.
- Compare neural networks with traditional baseline models.
- Identify effective preprocessing, architecture, and training choices.
- Measure the impact of regularization and optimization techniques.
- Deliver a reproducible pipeline for predicting booking cancellations.

---

## 11. References

1. [Hotel Booking Demand dataset on Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand/data)
2. Antonio, N., de Almeida, A., & Nunes, L. (2019). Hotel booking demand datasets. *Data in Brief, 22*, 41–49.
3. [TensorFlow Documentation](https://www.tensorflow.org/)
4. [Scikit-learn Documentation](https://scikit-learn.org/)

---

## 12. Project Status

**Current Phase:** Project setup and planning

Completed activities:

- [x] Project topic selected.
- [x] Dataset selected.
- [x] Problem statement defined.
- [x] Team roles assigned.
- [x] GitHub repository created.

Upcoming activities:

- [ ] Exploratory data analysis.
- [ ] Data preprocessing.
- [ ] Baseline-model implementation.
- [ ] Neural-network development.
- [ ] Model experimentation and evaluation.
- [ ] Final report and presentation.

---

**Course:** AIE231 – Neural Networks  
**University:** Alamein International University (AIU)  
**Project:** Hotel Booking Cancellation Prediction Using Neural Networks
