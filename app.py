import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# -----------------------------------------------------------
# Page setup
# -----------------------------------------------------------
st.set_page_config(page_title="Student Depression Risk", page_icon="🎓", layout="wide")

# -----------------------------------------------------------
# Load data and pre-trained artifacts (cached so it only runs once)
# -----------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("student_lifestyle_100k.csv")
    return df

@st.cache_resource
def load_artifacts():
    lr_model = joblib.load("Logistic_model.pkl")
    dt_model = joblib.load("decision_tree_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return lr_model, dt_model, scaler, feature_names

df = load_data()
lr_model, dt_model, scaler, feature_names = load_artifacts()

numerical_features = ["Age", "CGPA", "Sleep_Duration", "Study_Hours",
                       "Social_Media_Hours", "Physical_Activity", "Stress_Level"]

# Columns the scaler expects (matches the notebook's scaling step)
numerical_columns = ["Age", "CGPA", "Sleep_Duration", "Study_Hours", "Social_Media_Hours",
                      "Physical_Activity", "Stress_Level", "Sleep_Deficit", "Total_Daily_Load"]

# Hardcoded evaluation results, copied from the notebook (test set, no retraining here)
METRICS = {
    "Logistic Regression": {
        "accuracy": 0.6190, "precision": 0.1622, "recall": 0.6695, "f1": 0.2612, "roc_auc": 0.6796,
        "confusion_matrix": [[11032, 6956], [665, 1347]],
        "best_params": {"C": 1, "solver": "lbfgs"},
    },
    "Decision Tree": {
        "accuracy": 0.8162, "precision": 0.1510, "recall": 0.1789, "f1": 0.1638, "roc_auc": 0.5332,
        "confusion_matrix": [[15964, 2024], [1652, 360]],
        "best_params": {"criterion": "gini", "max_depth": None, "min_samples_leaf": 1, "min_samples_split": 2},
    },
}

# -----------------------------------------------------------
# Sidebar navigation
# -----------------------------------------------------------
st.sidebar.title("🎓 Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Exploratory Data Analysis", "Predict Depression Risk", "Model Performance"])

st.sidebar.markdown("---")
st.sidebar.caption("Dataset: Student Depression & Lifestyle (100k Data) — Kaggle")
st.sidebar.caption("Models: Logistic Regression & Decision Tree (pre-trained, not retrained here)")

# =============================================================
# PAGE 1 — OVERVIEW
# =============================================================
if page == "Overview":
    st.title("🎓 Student Lifestyle & Depression Risk")
    st.markdown(
        "This app explores lifestyle habits of **100,000 students** and predicts "
        "**depression risk** using two machine learning models trained earlier in a Jupyter notebook."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Students", f"{len(df):,}")
    col2.metric("Depression Cases", f"{int(df['Depression'].sum()):,}")
    col3.metric("Depression Rate", f"{df['Depression'].mean()*100:.2f}%")
    col4.metric("Features Used", len(feature_names))

    st.markdown("### What's inside this app")
    st.markdown(
        """
        - **Exploratory Data Analysis** — distributions, correlations, and patterns in the data
        - **Predict Depression Risk** — enter a student's lifestyle info and get a prediction
        - **Model Performance** — accuracy, precision, recall and confusion matrices for both models
        """
    )

    st.markdown("### A quick look at the data")
    st.dataframe(df.head(10), use_container_width=True)

# =============================================================
# PAGE 2 — EDA
# =============================================================
elif page == "Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis")

    # --- Target distribution ---
    st.subheader("1. Depression Distribution")
    col1, col2 = st.columns([2, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x="Depression", ax=ax)
        ax.set_title("Distribution of Depression")
        ax.set_xlabel("Depression")
        ax.set_ylabel("Count")
        st.pyplot(fig)
    with col2:
        pct = (df["Depression"].value_counts(normalize=True) * 100).round(2)
        st.write("**Class balance:**")
        st.write(f"- Not Depressed: {pct.get(False, 0)}%")
        st.write(f"- Depressed: {pct.get(True, 0)}%")
        st.info("The target is imbalanced: about 90% of students are not depressed, "
                "while only around 10% are. Accuracy alone won't tell the full story here, "
                "so recall and F1-score matter more.")

    st.markdown("---")

    # --- Numerical distributions ---
    st.subheader("2. Numerical Feature Distributions")
    selected_num = st.selectbox("Choose a numerical feature", numerical_features)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df[selected_num], bins=20, edgecolor="black")
    ax.set_title(f"Distribution of {selected_num}")
    ax.set_xlabel(selected_num)
    ax.set_ylabel("Count")
    st.pyplot(fig)
    st.caption("Sleep Duration is roughly normal around 7 hours. Study Hours and Social Media "
               "Hours are slightly right-skewed. Age and Physical Activity are spread fairly evenly.")

    st.markdown("---")

    # --- Categorical distributions ---
    st.subheader("3. Categorical Feature Distributions")
    cat_col = st.selectbox("Choose a categorical feature", ["Gender", "Department"])
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=df, x=cat_col, ax=ax)
    ax.set_title(f"Distribution of {cat_col}")
    plt.xticks(rotation=20)
    st.pyplot(fig)
    st.caption("Both Gender and Department are fairly balanced across categories, "
               "so no single group dominates the dataset.")

    st.markdown("---")

    # --- Correlation heatmap ---
    st.subheader("4. Correlation Heatmap")
    corr_df = df.copy()
    corr_df["Depression"] = corr_df["Depression"].astype(int)
    correlation_matrix = corr_df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)
    st.caption("Most features have a weak relationship with Depression individually. "
               "Physical Activity and Stress Level, and Sleep Duration and Stress Level, "
               "show the strongest relationships with each other.")

    st.markdown("---")

    # --- Boxplots by Depression ---
    st.subheader("5. Feature Comparison by Depression Status")
    box_col = st.selectbox("Compare which feature?", numerical_features, key="box_feat")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x="Depression", y=box_col, ax=ax)
    ax.set_title(f"{box_col} by Depression")
    st.pyplot(fig)

# =============================================================
# PAGE 3 — PREDICTION
# =============================================================
elif page == "Predict Depression Risk":
    st.title("🔮 Predict Depression Risk")
    st.markdown("Enter a student's lifestyle details below to estimate their depression risk. "
                "Predictions are made using the models trained earlier — no retraining happens here.")

    model_choice = st.radio("Choose a model", ["Logistic Regression", "Decision Tree"], horizontal=True)

    st.markdown("#### Student details")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Age", int(df["Age"].min()), int(df["Age"].max()), 21)
        gender = st.selectbox("Gender", ["Male", "Female"])
        department = st.selectbox("Department", ["Arts", "Business", "Engineering", "Medical", "Science"])
    with c2:
        cgpa = st.slider("CGPA", float(df["CGPA"].min()), float(df["CGPA"].max()), 3.0, step=0.01)
        sleep_duration = st.slider("Sleep Duration (hours/day)", 0.0, 12.0, 7.0, step=0.1)
        study_hours = st.slider("Study Hours (hours/day)", 0.0, 12.0, 4.0, step=0.1)
    with c3:
        social_media_hours = st.slider("Social Media Hours (hours/day)", 0.0, 12.0, 3.0, step=0.1)
        physical_activity = st.slider("Physical Activity (minutes/week)",
                                       int(df["Physical_Activity"].min()), int(df["Physical_Activity"].max()), 100)
        stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)

    if st.button("Predict", type="primary"):
        # Build a single-row dataframe matching the notebook's preprocessing
        input_dict = {
            "Age": age,
            "Gender": 0 if gender == "Male" else 1,
            "CGPA": cgpa,
            "Sleep_Duration": sleep_duration,
            "Study_Hours": study_hours,
            "Social_Media_Hours": social_media_hours,
            "Physical_Activity": physical_activity,
            "Stress_Level": stress_level,
            "Department_Business": 1 if department == "Business" else 0,
            "Department_Engineering": 1 if department == "Engineering" else 0,
            "Department_Medical": 1 if department == "Medical" else 0,
            "Department_Science": 1 if department == "Science" else 0,
        }
        input_df = pd.DataFrame([input_dict])

        # Feature engineering (same formulas as the notebook)
        input_df["Sleep_Deficit"] = (8 - input_df["Sleep_Duration"]).clip(lower=0)
        input_df["Total_Daily_Load"] = input_df["Study_Hours"] + input_df["Social_Media_Hours"]

        # Reorder columns to match training feature order exactly
        input_df = input_df[feature_names]

        # Scale numerical columns using the saved (already-fit) scaler
        input_scaled = input_df.copy()
        input_scaled[numerical_columns] = scaler.transform(input_df[numerical_columns])

        model = lr_model if model_choice == "Logistic Regression" else dt_model
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        st.markdown("---")
        st.subheader("Result")
        colA, colB = st.columns(2)
        with colA:
            if prediction == 1:
                st.error(f"⚠️ Predicted: **Depression risk detected**")
            else:
                st.success(f"✅ Predicted: **No depression risk detected**")
        with colB:
            st.metric("Estimated probability of depression", f"{probability*100:.1f}%")

        st.progress(min(max(probability, 0.0), 1.0))
        st.caption(f"Prediction made using: {model_choice}")
        st.warning("This is an educational classroom project, not a medical diagnostic tool. "
                   "Please don't use this for real mental health decisions.")

# =============================================================
# PAGE 4 — MODEL PERFORMANCE
# =============================================================
elif page == "Model Performance":
    st.title("📈 Model Performance")
    st.markdown("Both models were trained and tuned with `GridSearchCV` in the notebook "
                "(optimizing for recall, since catching depression cases matters most). "
                "The metrics below come straight from that evaluation on the held-out test set.")

    model_pick = st.selectbox("Select a model to inspect", list(METRICS.keys()))
    m = METRICS[model_pick]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{m['accuracy']*100:.2f}%")
    col2.metric("Precision", f"{m['precision']*100:.2f}%")
    col3.metric("Recall", f"{m['recall']*100:.2f}%")
    col4.metric("F1-score", f"{m['f1']*100:.2f}%")
    col5.metric("ROC-AUC", f"{m['roc_auc']:.3f}")

    st.markdown(f"**Best hyperparameters found:** `{m['best_params']}`")

    st.markdown("---")
    st.subheader("Confusion Matrix")
    cm = np.array(m["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Depressed", "Depressed"],
                yticklabels=["Not Depressed", "Depressed"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_pick}")
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Comparing both models")
    comp_df = pd.DataFrame({
        "Model": list(METRICS.keys()),
        "Accuracy": [METRICS[k]["accuracy"] for k in METRICS],
        "Precision": [METRICS[k]["precision"] for k in METRICS],
        "Recall": [METRICS[k]["recall"] for k in METRICS],
        "F1-score": [METRICS[k]["f1"] for k in METRICS],
        "ROC-AUC": [METRICS[k]["roc_auc"] for k in METRICS],
    })
    st.dataframe(comp_df.style.format({
        "Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}",
        "F1-score": "{:.4f}", "ROC-AUC": "{:.4f}"
    }), use_container_width=True)

    st.info("Decision Tree has higher accuracy, but Logistic Regression has much higher recall "
            "and F1-score. Since the goal is to catch students at risk of depression, "
            "recall matters more here — so Logistic Regression is the better choice for this task.")
