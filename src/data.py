"""Leakage-safe data loading, splitting, and preprocessing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "is_canceled"

# These fields are either outcome information or are not reliably available when
# a reservation is first created. See data/README.md for the availability audit.
LEAKAGE_COLUMNS = (
    "reservation_status",
    "reservation_status_date",
    "assigned_room_type",
    "booking_changes",
    "days_in_waiting_list",
    "required_car_parking_spaces",
    "total_of_special_requests",
)

# These columns are numeric identifiers, not continuous quantities.
CATEGORICAL_IDENTIFIER_COLUMNS = ("agent", "company")


@dataclass(frozen=True)
class DatasetSplits:
    """Raw, stratified train/validation/test data with no fitted transforms."""

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


def load_hotel_bookings(csv_path: str | Path) -> pd.DataFrame:
    """Load the Kaggle CSV and validate the target and required exclusions."""

    dataframe = pd.read_csv(csv_path, na_values=["NULL"])
    required = {TARGET_COLUMN, "reservation_status", "reservation_status_date"}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    target_values = set(dataframe[TARGET_COLUMN].dropna().unique())
    if not target_values.issubset({0, 1}):
        raise ValueError(f"{TARGET_COLUMN} must contain only 0 and 1; got {target_values}")
    return dataframe


def separate_features_target(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create a booking-creation feature matrix and binary cancellation target."""

    columns_to_remove = {TARGET_COLUMN, *LEAKAGE_COLUMNS}
    missing = columns_to_remove.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Cannot enforce feature policy; missing columns: {sorted(missing)}")

    X = dataframe.drop(columns=sorted(columns_to_remove)).copy()
    y = dataframe[TARGET_COLUMN].astype("int64").copy()

    # Preserve agent and company as categories rather than imposing an artificial
    # numeric order on their identifier values.
    for column in CATEGORICAL_IDENTIFIER_COLUMNS:
        if column in X.columns:
            X[column] = X[column].map(
                lambda value: str(int(value)) if pd.notna(value) else np.nan
            )

    forbidden_present = set(LEAKAGE_COLUMNS).intersection(X.columns)
    if forbidden_present:
        raise AssertionError(f"Leakage columns reached the feature matrix: {forbidden_present}")
    return X, y


def create_stratified_splits(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    random_state: int = 42,
    train_size: float = 0.70,
    validation_size: float = 0.15,
    test_size: float = 0.15,
) -> DatasetSplits:
    """Create 70/15/15 stratified splits without fitting any preprocessing."""

    if not np.isclose(train_size + validation_size + test_size, 1.0):
        raise ValueError("Split proportions must sum to 1.")

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=validation_size + test_size,
        stratify=y,
        random_state=random_state,
    )
    validation_fraction_of_holdout = validation_size / (validation_size + test_size)
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_holdout,
        y_holdout,
        test_size=1 - validation_fraction_of_holdout,
        stratify=y_holdout,
        random_state=random_state,
    )
    return DatasetSplits(X_train, X_validation, X_test, y_train, y_validation, y_test)


def infer_feature_columns(X_train: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return numeric and categorical feature names from training data only."""

    categorical = [
        column
        for column in X_train.columns
        if column in CATEGORICAL_IDENTIFIER_COLUMNS
        or not pd.api.types.is_numeric_dtype(X_train[column])
    ]
    numeric = [column for column in X_train.columns if column not in categorical]
    return numeric, categorical


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Build an unfitted transformer; callers must fit it on training data only."""

    numeric_columns, categorical_columns = infer_feature_columns(X_train)
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def split_class_distribution(splits: DatasetSplits) -> pd.DataFrame:
    """Summarize class counts and rates to verify stratification."""

    rows: list[dict[str, float | int | str]] = []
    for name, target in (
        ("train", splits.y_train),
        ("validation", splits.y_validation),
        ("test", splits.y_test),
    ):
        rows.append(
            {
                "split": name,
                "rows": len(target),
                "canceled_count": int(target.sum()),
                "canceled_rate": float(target.mean()),
            }
        )
    return pd.DataFrame(rows)
