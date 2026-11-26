# Hotel Booking Cancellation Prediction Using Neural Networks

**Course:** AIE231 – Neural Networks  
**University:** Alamein International University (AIU)  
**Faculty:** Faculty of Computer Science & Engineering

---

## 1. Project Overview

This project develops and evaluates neural-network models to predict whether a hotel reservation will be canceled before the guest arrives. Accurate cancellation predictions can help hotels improve inventory planning, reduce revenue loss, and make more informed operational decisions.

The project uses the [Hotel Booking Demand dataset](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand/data) and frames the task as a binary-classification problem.

### Project Objectives

- Analyze hotel booking data and identify patterns related to cancellations.
- Build a reliable data preprocessing pipeline.
- Prevent data leakage during data preparation and model evaluation.
- Establish machine learning baseline models.
- Develop and compare neural-network architectures in the later project stages.
- Evaluate model performance using appropriate classification metrics.

---

## 2. Problem Statement

Hotel cancellations create uncertainty in room availability, staffing, and revenue management. The goal is to build a model that estimates the likelihood that a booking will be canceled using reservation, guest, and booking-history information available before the booking outcome is known.

The target variable is `is_canceled`:

| Value | Meaning |
|---|---|
| `0` | The booking was not canceled. |
| `1` | The booking was canceled. |

This is a binary-classification task. The model's predictions can support targeted retention actions and more reliable room-demand planning.

### Business Impact

Accurate cancellation prediction can help hotels:

- Improve room inventory management.
- Make better booking and operational decisions.
- Identify bookings with a higher cancellation risk.
- Improve revenue planning and resource allocation.

Two types of prediction errors are important:

- **False Negative:** An actually canceled booking is predicted as not canceled. This may prevent the hotel from anticipating a cancellation.
- **False Positive:** A booking that is not canceled is incorrectly predicted as canceled. This may lead to unnecessary retention actions.

Both error types should be considered when evaluating the model.

---

## 3. Dataset Description

**Dataset Name:** Hotel Booking Demand

**Source:** Kaggle

**Dataset Link:**  
https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand/data

The dataset contains hotel-booking records for resort and city hotels. It combines numerical and categorical information about guests, reservation timing, stay characteristics, market segments, and previous cancellations.

### 3.1 Dataset Information

| Property | Description |
|---|---|
| Number of records | 119,390 |
| Total columns | 32 |
| Target variable | `is_canceled` |
| Problem type | Binary classification |
| Data types | Numerical and categorical |
| Hotel types | Resort hotel and city hotel |

### 3.2 Dataset Eligibility

The dataset satisfies the project's numerical size requirements:

- It contains 119,390 records, exceeding the minimum requirement of 10,000 records.
- It contains 32 columns, including the target variable, providing more than 10 potential input features.
- It represents a real-world hotel booking classification problem.
- It includes both numerical and categorical features.

The dataset's license and permitted academic usage should be checked on the original Kaggle dataset page before making any licensing claims.

### 3.3 Target Distribution

The target variable is `is_canceled`.

| Class | Number of Records | Percentage |
|---|---:|---:|
| Not Canceled (0) | 75,166 | 62.96% |
| Canceled (1) | 44,224 | 37.04% |
| **Total** | **119,390** | **100%** |

The target distribution indicates a moderate class imbalance. Therefore, accuracy alone is not sufficient to evaluate model performance.

### 3.4 Data Leakage Consideration

To ensure predictions use only information available before the booking outcome, the following columns are excluded from modeling:

- `reservation_status`
- `reservation_status_date`

Both fields reveal information about the final reservation outcome and would cause target leakage.

The implemented preprocessing pipeline also excludes additional conservative lifecycle or conditional fields for a booking-creation prediction setting. These exclusions are described in Section 6.

### 3.5 Dataset Challenges

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
- Investigate correlations and cancellation rates across categorical features.

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

The formula is:

\[
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
\]

### Additional Evaluation Metrics

- Accuracy
- Precision
- Recall
- Macro F1-score
- ROC-AUC
- Confusion matrix

These metrics provide a broader view of model performance, including the model's ability to identify canceled bookings and its overall classification behavior.

---

## 8. Week 10: EDA, Preprocessing, and Baselines

**Status: Completed**

The hotel EDA is available in `notebooks/01_eda.ipynb` and includes data-quality checks, class-balance analysis, distributions, IQR outlier flags, correlations, cancellation-rate comparisons, and a feature-availability audit.

### 8.1 Data Quality Findings

The EDA identified the following missing values:

| Feature | Missing Values |
|---|---:|
| company | 112,593 |
| agent | 16,340 |
| country | 488 |
| children | 4 |

The remaining inspected columns did not show missing values in the notebook's reported missing-value analysis.

### 8.2 Duplicate Records

The dataset contains **31,994 exact duplicate rows**.

These records were retained during the current analysis because the dataset does not provide a unique booking identifier that would allow us to distinguish legitimate identical bookings from accidental duplicate records.

This decision is documented and should be reconsidered if additional booking identifiers or source information become available.

### 8.3 EDA Analysis and Findings

The EDA notebook includes:

- Target distribution analysis.
- Numerical feature distributions.
- IQR-based outlier investigation.
- Correlation analysis.
- Categorical cancellation-rate comparisons.
- High-cardinality feature investigation.
- Missing-value and duplicate-record analysis.
- Feature-availability and leakage investigation.

These analyses informed the preprocessing and feature-selection decisions used in the baseline pipeline.

### 8.4 Leakage-Free Preprocessing

`src/data.py` implements the reusable data workflow:

- Separates `is_canceled` from model features.
- Always excludes `reservation_status` and `reservation_status_date`.
- Conservatively excludes `assigned_room_type`, `booking_changes`, `days_in_waiting_list`, `required_car_parking_spaces`, and `total_of_special_requests` for a booking-creation prediction setting.
- Imputes numerical features with the training-set median, then standardizes them.
- Imputes categorical features with `__MISSING__` and one-hot encodes them with unknown-category protection.
- Treats `agent` and `company` as categorical identifiers rather than continuous values.
- Creates stratified 70% / 15% / 15% train, validation, and test splits.

### 8.5 Data Splitting

The dataset was divided using stratified splitting:

| Dataset | Percentage | Number of Records |
|---|---:|---:|
| Training | 70% | 83,573 |
| Validation | 15% | 17,908 |
| Testing | 15% | 17,909 |
| **Total** | **100%** | **119,390** |

The cancellation rates were:

| Dataset | Cancellation Rate |
|---|---:|
| Training | 37.0419% |
| Validation | 37.0393% |
| Testing | 37.0428% |

Stratification preserves approximately the same target distribution across the three subsets.

### 8.6 Train-Only Preprocessing and Leakage Prevention

To prevent information leakage:

- The preprocessing pipeline is fitted using training data only.
- Validation data is transformed using the fitted training pipeline.
- Test data is transformed using the same fitted pipeline.
- Validation and test data are not used to fit preprocessing parameters.
- The test set is not used for model selection.

### 8.7 Validation Baselines

Two baseline models were implemented:

**1. Majority Classifier**

`DummyClassifier(strategy="most_frequent")`

This model always predicts the most frequent class and provides a simple reference point.

**2. Logistic Regression**

`LogisticRegression(class_weight="balanced", solver="liblinear")`

The balanced class-weight option helps account for the difference in class frequencies.

Both models were fitted on the training split and evaluated on the same validation split containing 17,908 rows, including 6,633 canceled bookings.

F1-score for `is_canceled = 1` is the primary metric.

### 8.8 Baseline Results

| Model | Accuracy | Precision (Canceled) | Recall (Canceled) | F1 (Canceled) | Macro F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Majority Classifier | 0.6296 | 0.0000 | 0.0000 | 0.0000 | 0.3864 | 0.5000 |
| Logistic Regression | 0.7911 | 0.7030 | 0.7550 | 0.7281 | 0.7792 | 0.8814 |

### 8.9 Confusion Matrices

The confusion matrices use the following layout:

- Rows represent actual classes.
- Columns represent predicted classes.

**Majority Classifier**

| | Predicted Not Canceled | Predicted Canceled |
|---|---:|---:|
| Actual Not Canceled | 11,275 | 0 |
| Actual Canceled | 6,633 | 0 |

**Logistic Regression**

| | Predicted Not Canceled | Predicted Canceled |
|---|---:|---:|
| Actual Not Canceled | 9,159 | 2,116 |
| Actual Canceled | 1,625 | 5,008 |

### 8.10 Baseline Interpretation

The Majority Classifier achieved 62.96% accuracy because it always predicted the majority class.

However, it failed to identify any canceled bookings, resulting in zero precision, recall, and F1-score for the canceled class.

Logistic Regression achieved:

- 79.11% accuracy.
- 70.30% precision for canceled bookings.
- 75.50% recall for canceled bookings.
- 72.81% F1-score for canceled bookings.
- 0.8814 ROC-AUC.

These results demonstrate that Logistic Regression provides a substantially stronger baseline than the majority-class prediction strategy.

It identifies a meaningful proportion of canceled bookings while maintaining a balance between precision and recall.

The test set was kept separate and was not used for baseline selection.

### 8.11 Reproduction

From the project root directory:

1. Download the Kaggle CSV to `data/raw/hotel_bookings.csv` as described in `data/README.md`.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the baseline pipeline:

```bash
python -m src.baselines --data data/raw/hotel_bookings.csv --output-dir experiments
```

4. Open the EDA notebook:

```bash
jupyter notebook
```

Then open `notebooks/01_eda.ipynb` and execute its cells.

The commands assume that the dependencies are installed and the dataset is available at the expected path.

---

## 9. Team Members and Responsibilities

| No. | Team Member | Role | Main Responsibilities |
|---|---|---|---|
| 1 | Yousif Emad Hassan | Tech Lead + EDA & Baselines | Project coordination, repository management, EDA, and baseline models |
| 2 | Judy Mohammed | Data Engineer | Data cleaning, preprocessing, encoding, and scaling |
| 3 | Ahmed Walid | Model Architect | Neural-network architecture design and implementation |
| 4 | Sara Saeed | Experimentation Engineer | Hyperparameter tuning, optimizer comparison, and ablation studies |
| 5 | Nadine Mohamed | Evaluation & Deployment Engineer | Model evaluation, error analysis, and inference pipeline |

All team members are expected to contribute to neural-network experimentation, documentation, and GitHub collaboration.

### Shared Responsibilities

- Commit individual contributions using personal GitHub accounts and feature branches.
- Submit changes through pull requests for review by another team member.
- Understand the complete project pipeline.
- Contribute to the final report and presentation.
- Explain technical decisions during the final discussion.

---

## 10. Project Structure

The active project repository is organized into the following main components:

```text
Hotel-Booking-Cancellation-NN/
├── .gitignore
├── README.md
├── requirements.txt
│
├── app/
├── configs/
│
├── data/
│   ├── README.md
│   └── raw/
│       └── hotel_bookings.csv  # Download separately; not committed
│
├── experiments/
│   ├── baseline_results.csv
│   ├── baseline_details.json
│   └── baseline_report.md
│
├── models/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── legacy/
│       ├── README.md
│       └── bank_marketing_eda.ipynb
│
├── report/
│
└── src/
    ├── __init__.py
    ├── data.py
    └── baselines.py
```

The active project uses `notebooks/01_eda.ipynb`. The prior Bank Marketing EDA is retained only as an archived reference under `notebooks/legacy/`.

The raw dataset is not committed to the repository.

---

## 11. Project Milestones and Progress

The project follows the suggested course timeline.

### Milestone 1 – Topic Settled (End of Week 9)

| Requirement | Status |
|---|---|
| Select a suitable real-world classification problem | Completed |
| Select the dataset | Completed |
| Review dataset size and features | Completed |
| Define the prediction target | Completed |
| Select the primary evaluation metric | Completed |
| Assign team responsibilities | Completed |
| Establish the GitHub repository | Completed |

### Milestone 2 – Baselines Ready (End of Week 10)

| Requirement | Status |
|---|---|
| Exploratory Data Analysis | Completed |
| Data quality investigation | Completed |
| Preprocessing pipeline | Completed |
| Stratified train/validation/test split | Completed |
| Leakage prevention | Completed |
| Majority Classifier baseline | Completed |
| Logistic Regression baseline | Completed |
| Validation evaluation | Completed |
| Baseline comparison and documentation | Completed |

### Upcoming Work

The following tasks belong to later project stages and are not claimed as completed:

- Neural-network architecture development.
- Training three neural-network models.
- Manual parameter-count calculations.
- Hyperparameter tuning and experiment tracking.
- Learning curve analysis.
- Model ablation studies.
- Final model selection.
- Test-set evaluation and error analysis.
- Inference pipeline and demonstration.

---

## 12. Expected Outcomes

- Develop multiple neural-network models for hotel-booking cancellation prediction.
- Compare neural networks with traditional baseline models.
- Identify effective preprocessing, architecture, and training choices.
- Measure the impact of regularization and optimization techniques.
- Deliver a reproducible pipeline for predicting booking cancellations.

---

## 13. References

1. [Hotel Booking Demand dataset on Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand/data)
2. Antonio, N., de Almeida, A., & Nunes, L. (2019). Hotel booking demand datasets. *Data in Brief, 22*, 41–49.
3. [TensorFlow Documentation](https://www.tensorflow.org/)
4. [Scikit-learn Documentation](https://scikit-learn.org/)
5. [DummyClassifier Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html)
6. [LogisticRegression Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

---

## 14. Project Status

**Current Phase:** Week 10 – Baselines Ready

### Completed Activities

- [x] Project topic selected.
- [x] Dataset selected.
- [x] Problem statement defined.
- [x] Team roles assigned.
- [x] GitHub repository created.
- [x] Hotel booking EDA completed.
- [x] Leakage-aware preprocessing pipeline implemented.
- [x] Stratified train/validation/test split implemented.
- [x] Majority-class baseline implemented and evaluated.
- [x] Logistic Regression baseline implemented and evaluated.
- [x] Baseline comparison documented.

### Upcoming Activities

- [ ] Neural-network development.
- [ ] Model experimentation and evaluation.
- [ ] Final report and presentation.

---

**Course:** AIE231 – Neural Networks  
**University:** Alamein International University (AIU)  
**Project:** Hotel Booking Cancellation Prediction Using Neural Networks

**Repository:**  
https://github.com/Yousif-jox/Hotel-Booking-Cancellation-NN
