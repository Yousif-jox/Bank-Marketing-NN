# Bank Marketing Subscription Prediction Using Neural Networks

## AIE231 – Neural Networks Project
**Alamein International University (AIU)**  
**Faculty of Computer Science and Engineering**

---

## 1. Project Overview

This project aims to develop and evaluate Neural Network models to predict whether a bank customer will subscribe to a term deposit as a result of a direct marketing campaign.

Banks often use direct marketing campaigns, such as phone calls, to promote their financial products. However, contacting every customer can be costly and inefficient.

By using machine learning and neural networks, this project attempts to identify customers who are more likely to subscribe, helping banks improve their marketing strategies and reduce unnecessary campaign costs.

The project uses the **Bank Marketing Dataset** from the UCI Machine Learning Repository and focuses on binary classification.

---

## 2. Problem Statement

Banks use marketing campaigns to encourage customers to subscribe to term deposits. However, contacting every customer can be costly and inefficient.

The main objective of this project is to develop a Neural Network model that predicts whether a customer will subscribe to a term deposit based on demographic information, financial details, and previous marketing campaign interactions.

The problem is a binary classification task, where the model predicts one of two classes:

- **Yes:** The customer subscribed to a term deposit.
- **No:** The customer did not subscribe to a term deposit.

These predictions can help banks improve their marketing strategies, reduce unnecessary campaign costs, and focus their efforts on customers who are more likely to subscribe.

Incorrect predictions may lead to wasted marketing resources or missed business opportunities.

---

## 3. Dataset Description

**Dataset Name:** Bank Marketing Dataset  
**Source:** UCI Machine Learning Repository  
**Dataset Link:** https://archive.ics.uci.edu/dataset/222/bank+marketing

### Dataset Information

- Number of records: 45,211
- Number of input features: 16
- Target variable: `y`
- Target classes: `yes` and `no`
- Problem type: Binary Classification
- Data types: Numerical and Categorical

### Target Variable

The target variable is `y`, which indicates whether a customer subscribed to a term deposit.

| Value | Meaning |
|---|---|
| yes | Customer subscribed |
| no | Customer did not subscribe |

### Dataset Challenges

The dataset includes several challenges that will be considered during preprocessing and model development:

- Imbalanced target classes.
- Numerical and categorical features.
- Unknown values in some categorical attributes.
- Different feature scales.
- Potential data leakage if certain features are used incorrectly.

### Data Leakage Consideration

The `duration` feature represents the duration of the last contact with the customer.

Since our goal is to predict customer subscription before the marketing call takes place, this feature will be excluded from the main prediction pipeline because it is not available at prediction time.

---

## 4. Project Objectives

The main objectives of this project are:

1. Understand and analyze the Bank Marketing Dataset.
2. Perform Exploratory Data Analysis (EDA).
3. Clean and preprocess the dataset.
4. Handle categorical and numerical features.
5. Build simple baseline classification models.
6. Develop different Neural Network architectures using TensorFlow/Keras.
7. Compare different optimizers and hyperparameters.
8. Apply regularization techniques to reduce overfitting.
9. Evaluate and compare model performance.
10. Analyze incorrect predictions and model limitations.
11. Develop a simple inference pipeline for making new predictions.

---

## 5. Technologies and Tools

The project will use the following technologies:

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Google Colab | Development environment |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Preprocessing and baseline models |
| TensorFlow / Keras | Neural Network development |
| Git & GitHub | Version control and collaboration |

---

## 6. Methodology

The project will follow these main stages:

### Stage 1: Exploratory Data Analysis (EDA)

- Inspect dataset structure and dimensions.
- Analyze numerical and categorical features.
- Examine target class distribution.
- Identify missing and unknown values.
- Visualize important relationships.

### Stage 2: Data Preprocessing

- Handle missing and unknown values.
- Encode categorical features.
- Scale numerical features.
- Split data into training, validation, and testing sets.
- Build a preprocessing pipeline to prevent data leakage.

### Stage 3: Baseline Models

The Neural Network models will be compared against simple baseline approaches:

- Majority Class Classifier.
- Logistic Regression.

### Stage 4: Neural Network Development

At least three Neural Network models will be developed:

1. Shallow Multi-Layer Perceptron (MLP).
2. Deeper MLP with regularization.
3. Best-performing MLP after hyperparameter tuning.

### Stage 5: Model Experimentation

- Compare SGD, SGD with Momentum, and Adam optimizers.
- Experiment with different learning rates.
- Apply Dropout and Early Stopping.
- Test different network architectures.
- Perform ablation studies to understand the contribution of different components.

### Stage 6: Model Evaluation

Evaluate the models using multiple classification metrics and compare their performance.

### Stage 7: Inference and Deployment

Create a prediction pipeline that accepts new customer information and returns a predicted subscription class.

---

## 7. Evaluation Metrics

Because the dataset contains imbalanced target classes, accuracy alone may not provide a complete picture of model performance.

### Primary Evaluation Metric

**F1-score for the positive class (`yes`).**

F1-score balances Precision and Recall and helps evaluate how effectively the model identifies customers who actually subscribe.

### Additional Evaluation Metrics

- Accuracy
- Precision
- Recall
- Macro-F1 Score
- ROC-AUC
- Confusion Matrix

All final models will be evaluated using the same test set for a fair comparison.

---

## 8. Team Members and Responsibilities

The project team consists of five members:

| No. | Team Member | Role | Main Responsibilities |
|---|---|---|---|
| 1 | Yousif Emad Hassan | Tech Lead + EDA & Baselines | Project coordination, repository management, EDA, and baseline models |
| 2 | Judy Mohammed | Data Engineer | Data cleaning, preprocessing, encoding, and scaling |
| 3 | Ahmed Walid | Model Architect | Neural Network architecture design and implementation |
| 4 | Sara Saeed | Experimentation Engineer | Hyperparameter tuning, optimizer comparison, and ablation studies |
| 5 | Nadine Mohamed | Evaluation & Deployment Engineer | Model evaluation, error analysis, and inference pipeline |

All team members will contribute to Neural Network experimentation, documentation, and GitHub collaboration.

---

## 9. Project Structure

The repository is organized as follows:

```text
Bank-Marketing-NN/
│
├── README.md
├── requirements.txt
│
├── configs/
│
├── data/
│   └── README.md
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_results.ipynb
│
├── src/
│   ├── data.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── experiments/
│
├── models/
│
├── app/
│
└── report/
```

---

## 10. Expected Outcomes

By the end of this project, we expect to:

- Develop multiple Neural Network models for bank marketing prediction.
- Compare Neural Networks against traditional baseline models.
- Identify the most effective model architecture and training configuration.
- Understand the effects of preprocessing, regularization, and optimization techniques.
- Evaluate the model's ability to identify potential term deposit subscribers.
- Deliver a reproducible prediction pipeline.

---

## 11. References

1. Moro, S., Cortez, P., & Rita, P. (2014). A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems.
2. UCI Machine Learning Repository – Bank Marketing Dataset: https://archive.ics.uci.edu/dataset/222/bank+marketing
3. TensorFlow Documentation: https://www.tensorflow.org/
4. Scikit-learn Documentation: https://scikit-learn.org/

---

## 12. Project Status

**Current Phase:** Week 9 – Project Planning

Completed planning activities:

- [Done] Project topic selected.
- [Done] Dataset selected.
- [Done] Problem statement defined.
- [Done] Team roles assigned.
- [Done] GitHub repository created.

Upcoming activities:

- [ ] Exploratory Data Analysis (EDA).
- [ ] Data preprocessing.
- [ ] Baseline model implementation.
- [ ] Neural Network development.
- [ ] Model experimentation and evaluation.
- [ ] Final report and presentation.

---

**Course:** AIE231 – Neural Networks  
**University:** Alamein International University (AIU)  
**Project:** Bank Marketing Subscription Prediction Using Neural Networks