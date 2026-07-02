"""
Data utilities for the Student Lifestyle & Depression Insights app.

The original project used a 100,000-row Kaggle dataset that isn't bundled
with this app. To keep the app fully self-contained and instantly runnable,
we regenerate a synthetic dataset that matches the real dataset's reported
statistics (means, std devs, ranges, class balance, and key correlations
from the EDA notebook). Swap in the real CSV any time via the sidebar
uploader and every chart/metric updates automatically.
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score
)

NUMERICAL_FEATURES = [
    "Age", "CGPA", "Sleep_Duration", "Study_Hours",
    "Social_Media_Hours", "Physical_Activity", "Stress_Level"
]
DEPARTMENTS = ["Science", "Engineering", "Medical", "Arts", "Business"]

# Real results reported in the source notebook (default-parameter models,
# trained without PCA) -- shown as a reference/benchmark in the app.
BENCHMARK_RESULTS = pd.DataFrame({
    "Model": [
        "Logistic Regression", "Decision Tree", "Random Forest",
        "KNN", "SVM", "Improved Logistic Regression (final)"
    ],
    "Accuracy": [0.8994, 0.8162, 0.8993, 0.8914, 0.8994, 0.6190],
    "Precision": [0.0000, 0.1510, 0.4824, 0.2846, 0.0000, 0.1622],
    "Recall": [0.0000, 0.1789, 0.0204, 0.0522, 0.0000, 0.6695],
    "F1-score": [0.0000, 0.1638, 0.0391, 0.0882, 0.0000, 0.2612],
})
FINAL_ROC_AUC = 0.6796


@st.cache_data(show_spinner=False)
def generate_synthetic_data(n=6000, seed=42):
    """Generate a synthetic sample that mirrors the real dataset's stats."""
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 25, n)
    gender = rng.choice(["Male", "Female"], n, p=[0.5012, 0.4988])
    department = rng.choice(DEPARTMENTS, n, p=[0.2007, 0.2006, 0.1996, 0.2000, 0.1991])
    cgpa = np.clip(rng.normal(2.898, 0.532, n), 1.56, 4.0)

    # Correlated lifestyle block: stress relates negatively to sleep & activity
    stress_raw = rng.normal(0, 1, n)
    sleep = np.clip(7.0 - 0.55 * stress_raw + rng.normal(0, 1.05, n), 3, 12)
    activity = np.clip(74.35 - 14 * stress_raw + rng.normal(0, 36, n), 0, 149)
    stress = np.clip(np.round(4.13 + 1.05 * stress_raw), 2, 10)

    study_hours = np.clip(rng.normal(4.51, 1.98, n), 0, 12.8)
    social_hours = np.clip(rng.normal(3.50, 1.49, n), 0, 10)

    sleep_deficit = np.clip(8 - sleep, 0, None)
    total_load = study_hours + social_hours

    # Depression risk: mildly driven by stress/sleep deficit/social load,
    # calibrated to land near the real 10.06% positive rate, plus noise
    # (mirrors the notebook's finding that features are only weakly related
    # to the target).
    risk_score = (
        0.35 * (stress - 4.13) / 1.42
        + 0.20 * (sleep_deficit - 1.0) / 1.3
        + 0.15 * (total_load - 8.0) / 2.4
        - 0.10 * (activity - 74) / 43
        + rng.normal(0, 1, n) * 1.35
    )
    threshold = np.quantile(risk_score, 0.8994)
    depression = risk_score > threshold

    df = pd.DataFrame({
        "Age": age,
        "Gender": gender,
        "Department": department,
        "CGPA": cgpa.round(2),
        "Sleep_Duration": sleep.round(1),
        "Study_Hours": study_hours.round(1),
        "Social_Media_Hours": social_hours.round(1),
        "Physical_Activity": activity.round(0).astype(int),
        "Stress_Level": stress.astype(int),
        "Sleep_Deficit": sleep_deficit.round(1),
        "Total_Daily_Load": total_load.round(1),
        "Depression": depression,
    })
    return df


@st.cache_resource(show_spinner=False)
def train_demo_model(df):
    """Train a small live logistic-regression model for the Risk Estimator.

    This mirrors the notebook's final approach (class-balanced Logistic
    Regression) but is trained fresh on the in-app synthetic sample, so the
    estimator is illustrative rather than the notebook's original model.
    """
    work = df.copy()
    work["Gender_Female"] = (work["Gender"] == "Female").astype(int)
    work = pd.get_dummies(work, columns=["Department"], drop_first=True, dtype=int)

    feature_cols = [
        "Age", "Gender_Female", "CGPA", "Sleep_Duration", "Study_Hours",
        "Social_Media_Hours", "Physical_Activity", "Stress_Level",
        "Sleep_Deficit", "Total_Daily_Load"
    ] + [c for c in work.columns if c.startswith("Department_")]

    X = work[feature_cols]
    y = work["Depression"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    num_cols = ["Age", "CGPA", "Sleep_Duration", "Study_Hours",
                "Social_Media_Hours", "Physical_Activity", "Stress_Level",
                "Sleep_Deficit", "Total_Daily_Load"]
    X_train_s, X_test_s = X_train.copy(), X_test.copy()
    X_train_s[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_s[num_cols] = scaler.transform(X_test[num_cols])

    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "roc_curve": roc_curve(y_test, y_prob),
    }
    return model, scaler, feature_cols, num_cols, metrics
