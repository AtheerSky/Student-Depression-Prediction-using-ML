import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# =============================================================
# PAGE SETUP
# =============================================================
st.set_page_config(page_title="Student Depression Risk", page_icon="🎓", layout="wide")

# -----------------------------------------------------------
# Design system — one consistent visual language for every page
# -----------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; color: #1F2937; }

/* section eyebrow label */
.eyebrow {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #0F766E;
    margin-bottom: 4px;
}

/* stat card grid */
.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 14px; margin: 10px 0 22px 0; }
.stat-card { background: #FFFFFF; border: 1px solid #E5E9F0; border-radius: 14px; padding: 16px 20px; }
.stat-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.07em; color: #6B7280; margin-bottom: 6px; }
.stat-value { font-family: 'Space Grotesk', sans-serif; font-size: 1.55rem; font-weight: 700; color: #1F2937; }
.stat-note { font-size: 0.76rem; color: #6B7280; margin-top: 3px; }

/* section divider */
.section-rule { border: none; border-top: 1px solid #E5E9F0; margin: 26px 0 18px 0; }

/* insight callout */
.insight-box { background: #F0F9F8; border-left: 3px solid #0F766E; border-radius: 8px; padding: 12px 16px; font-size: 0.92rem; color: #1F2937; margin-top: 10px; }
.conclusion-box { background: #FFF8ED; border-left: 3px solid #F2A541; border-radius: 8px; padding: 14px 18px; font-size: 0.94rem; color: #1F2937; margin-top: 12px; }

/* risk result card */
.risk-card { border-radius: 18px; padding: 26px 30px; text-align: center; border: 1.5px solid #E5E9F0; background: #FFFFFF; }
.risk-card.high { box-shadow: inset 0 0 0 2px #DC5B3E; }
.risk-card.low { box-shadow: inset 0 0 0 2px #0F766E; }
.risk-pill { display: inline-block; font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.25rem; padding: 6px 20px; border-radius: 999px; margin-bottom: 14px; }
.risk-pill.high { background: #FDEDE8; color: #C0402A; }
.risk-pill.low { background: #E7F5F3; color: #0F766E; }
.risk-percent { font-family: 'Space Grotesk', sans-serif; font-size: 2.6rem; font-weight: 700; color: #1F2937; margin-bottom: 16px; }
.risk-bar-track { position: relative; height: 12px; border-radius: 10px; background: linear-gradient(90deg, #0F766E, #F2A541, #DC5B3E); margin: 6px 6px 8px 6px; }
.risk-bar-marker { position: absolute; top: -6px; width: 4px; height: 24px; background: #1F2937; border-radius: 2px; }
.risk-scale { display: flex; justify-content: space-between; font-size: 0.72rem; color: #6B7280; padding: 0 6px; }
.risk-footer { font-size: 0.82rem; color: #6B7280; margin-top: 16px; }

/* best-model tag used in tables */
.badge-final { background:#E7F5F3; color:#0F766E; padding:2px 9px; border-radius:6px; font-size:0.75rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)


def eyebrow(text):
    st.markdown(f'<div class="eyebrow">{text}</div>', unsafe_allow_html=True)


def stat_grid(items):
    """items: list of (label, value, note-or-None)"""
    html = '<div class="stat-grid">'
    for label, value, note in items:
        note_html = f'<div class="stat-note">{note}</div>' if note else ""
        html += f'<div class="stat-card"><div class="stat-label">{label}</div><div class="stat-value">{value}</div>{note_html}</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def section_rule():
    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)


# =============================================================
# LOAD DATA & PRE-TRAINED ARTIFACTS (cached, never retrained)
# =============================================================
@st.cache_data
def load_data():
    return pd.read_csv("student_lifestyle_100k.csv")


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

numerical_columns = ["Age", "CGPA", "Sleep_Duration", "Study_Hours", "Social_Media_Hours",
                      "Physical_Activity", "Stress_Level", "Sleep_Deficit", "Total_Daily_Load"]

# -----------------------------------------------------------
# Hardcoded results copied from the notebook (test set, 20,000 rows).
# Nothing here is retrained inside the app.
# -----------------------------------------------------------

# Final, tuned models (GridSearchCV, cv=5, scoring="recall") — the two saved .pkl models
FINAL_METRICS = {
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

# Default (untuned) models — PCA vs. No PCA, matching notebook section 6
ALL_MODELS = [
    {"model": "Logistic Regression", "pca": "With PCA", "accuracy": 0.8994, "precision": 0.0000, "recall": 0.0000, "f1": 0.0000},
    {"model": "Decision Tree", "pca": "With PCA", "accuracy": 0.8216, "precision": 0.1689, "recall": 0.1973, "f1": 0.1820},
    {"model": "Random Forest", "pca": "With PCA", "accuracy": 0.8992, "precision": 0.4667, "recall": 0.0174, "f1": 0.0335},
    {"model": "KNN", "pca": "With PCA", "accuracy": 0.8892, "precision": 0.2437, "recall": 0.0482, "f1": 0.0805},
    {"model": "SVM", "pca": "With PCA", "accuracy": 0.8994, "precision": 0.0000, "recall": 0.0000, "f1": 0.0000},
    {"model": "Logistic Regression", "pca": "Without PCA", "accuracy": 0.8994, "precision": 0.0000, "recall": 0.0000, "f1": 0.0000},
    {"model": "Decision Tree", "pca": "Without PCA", "accuracy": 0.8162, "precision": 0.1510, "recall": 0.1789, "f1": 0.1638},
    {"model": "Random Forest", "pca": "Without PCA", "accuracy": 0.8992, "precision": 0.4824, "recall": 0.0204, "f1": 0.0391},
    {"model": "KNN", "pca": "Without PCA", "accuracy": 0.8914, "precision": 0.2846, "recall": 0.0522, "f1": 0.0882},
    {"model": "SVM", "pca": "Without PCA", "accuracy": 0.8994, "precision": 0.0000, "recall": 0.0000, "f1": 0.0000},
]

PCA_INFO = {
    "original_features": 14,
    "n_components": 9,
    "cumulative_variance": 0.9504,
}

# =============================================================
# SIDEBAR NAVIGATION
# =============================================================
st.sidebar.title("🎓Student Depression Prediction")
page = st.sidebar.radio("Menu", [
    "Overview",
    "Exploratory Data Analysis",
    "Model Comparison",
    "Final Model Evaluation",
    "Predict Depression Risk",
])

st.sidebar.markdown("---")
st.sidebar.caption("Dataset: Student Depression & Lifestyle (100k Data)")
st.sidebar.caption("Final model: Logistic Regression (Improved)")
st.sidebar.caption("Atheer Almajnoni")


# =============================================================
# PAGE 1 — OVERVIEW
# =============================================================
if page == "Overview":
    eyebrow("Student Lifestyle & Mental Health")
    st.title("🎓 Depression Risk in Student Life")
    st.markdown(
        "This app explores lifestyle habits of **100,000 students** and estimates "
        "**depression risk** using machine learning models trained and tuned in a companion notebook."
    )

    stat_grid([
        ("Total Students", f"{len(df):,}", None),
        ("Depression Cases", f"{int(df['Depression'].sum()):,}", None),
        ("Depression Rate", f"{df['Depression'].mean()*100:.2f}%", "imbalanced target"),
        ("Features Used", f"{len(feature_names)}", "after encoding & feature engineering"),
    ])

    section_rule()
    eyebrow("How this app is organized")
    st.markdown(
        """
        - **Exploratory Data Analysis** — distributions, correlations, and patterns in the raw data
        - **Model Comparison** — all five candidate models, with and without PCA
        - **Final Model Evaluation** — the two tuned models that were actually saved and used here
        - **Predict Depression Risk** — enter a student's details and get a live risk estimate
        """
    )

    section_rule()
    eyebrow("Raw data")
    st.dataframe(df.head(10), use_container_width=True)

# =============================================================
# PAGE 2 — EDA
# =============================================================
elif page == "Exploratory Data Analysis":
    eyebrow("Section 02")
    st.title("📊 Exploratory Data Analysis")

    st.subheader("1. Depression Distribution")
    col1, col2 = st.columns([2, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x="Depression", palette=["#0F766E", "#DC5B3E"], ax=ax)
        ax.set_title("Distribution of Depression")
        ax.set_xlabel("Depression")
        ax.set_ylabel("Count")
        st.pyplot(fig)
    with col2:
        pct = (df["Depression"].value_counts(normalize=True) * 100).round(2)
        st.markdown("**Class balance:**")
        st.write(f"- Not Depressed: {pct.get(False, 0)}%")
        st.write(f"- Depressed: {pct.get(True, 0)}%")
        st.markdown(
            '<div class="insight-box">The target is imbalanced: about 90% of students are not '
            'depressed, while only around 10% are. Accuracy alone won\'t tell the full story here, '
            'so recall and F1-score matter more.</div>', unsafe_allow_html=True)

    section_rule()

    st.subheader("2. Numerical Feature Distributions")
    selected_num = st.selectbox("Select a feature to explore", numerical_features)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df[selected_num], bins=20, edgecolor="black", color="#0F766E")
    ax.set_title(f"Distribution of {selected_num}")
    ax.set_xlabel(selected_num)
    ax.set_ylabel("Count")
    st.pyplot(fig)
    

    if selected_num == "Age":
        st.caption("Most students are between 18 and 24 years old, with ages fairly evenly distributed.")
    elif selected_num == "Sleep_Duration":
        st.caption("Most students sleep around 7 hours per night, with fewer students sleeping very little or very long.")
    elif selected_num == "Study_Hours":
        st.caption("Most students study a moderate number of hours each day, while fewer study for very long periods.")
    elif selected_num == "Social_Media_Hours":
        st.caption("Social media usage is slightly right-skewed, meaning a small number of students spend many hours online.")
    elif selected_num == "Physical_Activity":
        st.caption("Physical activity is fairly spread across the dataset, showing different activity levels among students.")
    elif selected_num == "CGPA":
        st.caption("CGPA values are concentrated around the middle range, with fewer students at the lowest and highest values.")
    elif selected_num == "Stress_Level":
        st.caption("Stress levels vary across students, with most reporting moderate stress.")
    
    section_rule()

    st.subheader("3. Categorical Feature Distributions")
    cat_col = st.selectbox("Choose a categorical feature", ["Gender", "Department"])
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=df, x=cat_col, color="#0F766E", ax=ax)
    ax.set_title(f"Distribution of {cat_col}")
    plt.xticks(rotation=20)
    st.pyplot(fig)
    st.caption("Both Gender and Department are fairly balanced across categories, "
               "so no single group dominates the dataset.")

    section_rule()

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

    section_rule()

    st.subheader("5. Feature Comparison by Depression Status")
    box_col = st.selectbox("Compare which feature?", numerical_features, key="box_feat")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x="Depression", y=box_col, palette=["#0F766E", "#DC5B3E"], ax=ax)
    ax.set_title(f"{box_col} by Depression")
    st.pyplot(fig)

# =============================================================
# PAGE 3 — MODEL COMPARISON (all default models + PCA vs No PCA)
# =============================================================
elif page == "Model Comparison":
    eyebrow("Section 06 · Modeling")
    st.title("🧪 Model Comparison")
    st.markdown(
        "Before the models' optimisztion, five classification algorithms were trained using their **default parameters**. " 
                "Also, each model was evaluated with and without **Principal Component Analysis (PCA)** to compare the effect of dimensionality reduction on classification performance."
    )

    comp_df = pd.DataFrame(ALL_MODELS)[["model", "pca", "accuracy", "precision", "recall", "f1"]]
    comp_df.columns = ["Model", "PCA", "Accuracy", "Precision", "Recall", "F1-score"]
    comp_df = comp_df.sort_values(by="F1-score", ascending=False).reset_index(drop=True)

    st.dataframe(
        comp_df.style.format({
            "Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}", "F1-score": "{:.4f}"
        }).background_gradient(subset=["F1-score"], cmap="BuGn"),
        use_container_width=True,
    )
    st.caption("Models are sorted by F1-score. Because the dataset is imbalanced (~90:10), accuracy alone can be misleading." 
               " A model may achieve high accuracy by correctly predicting the majority class (students without depression), while failing to identify many students with depression."
               " while the F1-score provides a more balanced evaluation of model performance.")

    section_rule()
    eyebrow("Dimensionality Reduction")
    st.subheader("PCA vs. No PCA")

    stat_grid([
        ("Original Features", PCA_INFO["original_features"], None),
        ("Components for 95% Variance", PCA_INFO["n_components"], None),
        ("Variance Retained", f"{PCA_INFO['cumulative_variance']*100:.2f}%", None),
    ])

    st.markdown(
        '<div class="conclusion-box"><b>Conclusion:</b>  PCA reduced the number of features from 14 to 9'
        'while preserving approximately 95% of the original variance.'
        'Based on the comparison, the Decision Tree achieved <b>better</b> performance with PCA, while KNN performed <b>better without </b> PCA.'
        'Logistic Regression, Random Forest, and SVM produced very similar results in both cases.'
        
        'The correlation heatmap indicates that the dataset contains generally weak correlations between features,'
        'which may explain why PCA had only a limited impact on the performance of most models.'
        'of most models.</div>', unsafe_allow_html=True)






# =============================================================
# PAGE 4 — FINAL MODEL EVALUATION
# =============================================================
elif page == "Final Model Evaluation":
    eyebrow("Section 07 · Evaluation")
    st.title("📈 Final Model Evaluation")
    st.markdown( "The two best-performing models, **Logistic Regression and Decision Tree**," 
                 "were optimized using **GridSearchCV with 5-fold cross-validation**. " 
                 "**Recall** was selected as the optimization metric because the dataset is imbalanced and identifying students at risk of depression is the primary objective." )

    model_pick = st.selectbox("Select a model to inspect", list(FINAL_METRICS.keys()))
    m = FINAL_METRICS[model_pick]

    stat_grid([
        ("Accuracy", f"{m['accuracy']*100:.2f}%", None),
        ("Precision", f"{m['precision']*100:.2f}%", None),
        ("Recall", f"{m['recall']*100:.2f}%", None),
        ("F1-score", f"{m['f1']*100:.2f}%", None),
        ("ROC-AUC", f"{m['roc_auc']:.3f}", None),
    ])
    st.markdown(f"**Best hyperparameters found:** `{m['best_params']}`")

    section_rule()
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

    section_rule()
    st.subheader("Improved Logistic Regression vs. Improved Decision Tree")
    improved_df = pd.DataFrame({
        "Model": list(FINAL_METRICS.keys()),
        "Accuracy": [FINAL_METRICS[k]["accuracy"] for k in FINAL_METRICS],
        "Precision": [FINAL_METRICS[k]["precision"] for k in FINAL_METRICS],
        "Recall": [FINAL_METRICS[k]["recall"] for k in FINAL_METRICS],
        "F1-score": [FINAL_METRICS[k]["f1"] for k in FINAL_METRICS],
        "ROC-AUC": [FINAL_METRICS[k]["roc_auc"] for k in FINAL_METRICS],
    })
    st.dataframe(improved_df.style.format({
        "Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}",
        "F1-score": "{:.4f}", "ROC-AUC": "{:.4f}"
    }), use_container_width=True)

    st.markdown(
        '<div class="conclusion-box">'
        '<b>Why Logistic Regression was chosen as the final model ?</b> '
        '<span class="badge-final">FINAL MODEL</span><br><br>'
        'The Decision Tree achieved higher accuracy, but accuracy alone is not enough because the dataset is imbalanced.'
        'The main goal of this project is to identify as many students with depression as possible.'
        
        '   The improved Logistic Regression achieved a much higher <b>Recall (66.95%) and F1-score (26.12%)</b> than the Decision Tree.'
        'It also achieved a higher <b>ROC-AUC (0.680)<b>'   
        'a false alarm.    The tuned Logistic Regression reaches <b>66.95% recall</b> and an F1-score '
        'Therefore, the improved Logistic Regression was chosen as the final model for prediction.'
        '</div>', unsafe_allow_html=True
    )




# =============================================================
# PAGE 5 — PREDICTION
# =============================================================
elif page == "Predict Depression Risk":
    eyebrow("Live Inference")
    st.title("🔮 Predict Depression Risk")
    st.markdown("Enter a student's lifestyle details below to estimate their depression risk. ")

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
        input_df["Sleep_Deficit"] = (8 - input_df["Sleep_Duration"]).clip(lower=0)
        input_df["Total_Daily_Load"] = input_df["Study_Hours"] + input_df["Social_Media_Hours"]
        input_df = input_df[feature_names]

        input_scaled = input_df.copy()
        input_scaled[numerical_columns] = scaler.transform(input_df[numerical_columns])

        model = lr_model if model_choice == "Logistic Regression" else dt_model
        prediction = model.predict(input_scaled)[0]
        probability = float(model.predict_proba(input_scaled)[0][1])
        pct = probability * 100

        risk_class = "high" if prediction == 1 else "low"
        pill_text = "⚠️ HIGH RISK" if risk_class == "high" else "✅ LOW RISK"

        section_rule()
        st.subheader("Result")

        st.markdown(f"""
        <div class="risk-card {risk_class}">
            <div class="risk-pill {risk_class}">{pill_text}</div>
            <div class="risk-percent">{pct:.1f}%</div>
            <div style="color:#6B7280; font-size:0.85rem; margin-bottom:10px;">estimated probability of depression</div>
            <div class="risk-bar-track">
                <div class="risk-bar-marker" style="left: calc({min(max(pct,0),100)}% - 2px);"></div>
            </div>
            <div class="risk-scale"><span>Low</span><span>Moderate</span><span>High</span></div>
            <div class="risk-footer">Prediction made using: {model_choice}</div>
        </div>
        """, unsafe_allow_html=True)

        st.warning("This application is for educational purposes only and is not intended for medical diagnosis." )
