"""
======================================================================
 Student Depression Risk Predictor
 A Streamlit app built on top of an already-trained ML pipeline
 (Project by Atheer)
======================================================================
This app does NOT train any models. All models were trained,
tuned (GridSearchCV) and evaluated inside the Jupyter notebook.
Here we only LOAD the saved artifacts (scaler, feature list, and the
two final models) and use them to explore the data and make
predictions.
======================================================================
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------
# 1. PAGE CONFIG  +  CUSTOM STYLE
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Student Depression Risk Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# A calm, warm colour palette (teal + sand + soft coral) instead of
# harsh red/green, since the topic is mental health.
PRIMARY = "#2A9D8F"      # teal
PRIMARY_DARK = "#1D6F64"
ACCENT = "#E9C46A"       # warm sand
WARN = "#E76F51"         # soft coral (used instead of alarming red)
INK = "#264653"          # deep navy text
CARD_BG = "#FFFFFF"
PAGE_BG = "#F6FAF9"

st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: {PAGE_BG};
        }}
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {PRIMARY_DARK} 0%, {PRIMARY} 100%);
        }}
        section[data-testid="stSidebar"] * {{
            color: #F6FAF9 !important;
        }}
        section[data-testid="stSidebar"] .stRadio > label {{
            font-weight: 600;
        }}
        h1, h2, h3 {{
            color: {INK};
            font-family: "Trebuchet MS", sans-serif;
        }}
        .hero {{
            background: linear-gradient(120deg, {PRIMARY} 0%, {PRIMARY_DARK} 100%);
            padding: 2rem 2.2rem;
            border-radius: 18px;
            color: white;
            margin-bottom: 1.5rem;
        }}
        .hero h1 {{
            color: white !important;
            margin-bottom: 0.3rem;
        }}
        .hero p {{
            color: #EAF6F3;
            font-size: 1.05rem;
        }}
        .metric-card {{
            background-color: {CARD_BG};
            border-radius: 14px;
            padding: 1.1rem 1.3rem;
            border: 1px solid #E3EFEC;
            box-shadow: 0 2px 10px rgba(38,70,83,0.05);
        }}
        .badge-low {{
            background-color: #E7F6EE;
            color: #1D7A46;
            padding: 0.9rem 1.2rem;
            border-radius: 14px;
            font-size: 1.15rem;
            font-weight: 700;
            border: 1px solid #BCE7CE;
        }}
        .badge-high {{
            background-color: #FDEDE8;
            color: #B8412A;
            padding: 0.9rem 1.2rem;
            border-radius: 14px;
            font-size: 1.15rem;
            font-weight: 700;
            border: 1px solid #F5C6B8;
        }}
        .footnote {{
            color: #6B7A78;
            font-size: 0.85rem;
        }}
        div[data-testid="stMetricValue"] {{
            color: {PRIMARY_DARK};
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# 2. LOAD SAVED ARTIFACTS  (cached so they only load once)
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    """Load the scaler, feature list and the two trained models."""
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    log_model = joblib.load("Logistic_model.pkl")
    tree_model = joblib.load("decision_tree_model.pkl")
    return scaler, feature_names, log_model, tree_model


@st.cache_data
def load_dataset():
    """Load the raw dataset used for the EDA page."""
    df = pd.read_csv("student_lifestyle_100k.csv")
    return df


scaler, FEATURE_NAMES, log_model, tree_model = load_artifacts()
raw_df = load_dataset()

# Columns that were scaled in the notebook (must match training exactly)
NUMERICAL_COLUMNS = [
    "Age", "CGPA", "Sleep_Duration", "Study_Hours", "Social_Media_Hours",
    "Physical_Activity", "Stress_Level", "Sleep_Deficit", "Total_Daily_Load",
]
DEPARTMENTS = ["Arts", "Business", "Engineering", "Medical", "Science"]


# ----------------------------------------------------------------------
# 3. PREPROCESSING HELPERS
#    (mirrors the exact cleaning / encoding / feature-engineering
#     steps from the notebook -- no model is ever re-trained here)
# ----------------------------------------------------------------------
def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduces the notebook's cleaning + encoding + feature engineering."""
    df_clean = df.drop(columns="Student_ID").copy()

    # Outlier capping (IQR method) - same 4 columns as the notebook
    for col in ["Sleep_Duration", "Study_Hours", "Social_Media_Hours", "Stress_Level"]:
        q1, q3 = df_clean[col].quantile(0.25), df_clean[col].quantile(0.75)
        iqr = q3 - q1
        df_clean[col] = df_clean[col].clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)

    df_clean["Depression"] = df_clean["Depression"].astype(int)
    df_clean["Gender"] = df_clean["Gender"].map({"Male": 0, "Female": 1})
    df_clean = pd.get_dummies(df_clean, columns=["Department"], drop_first=True, dtype=int)

    # Feature engineering
    df_clean["Sleep_Deficit"] = (8 - df_clean["Sleep_Duration"]).clip(lower=0)
    df_clean["Total_Daily_Load"] = df_clean["Study_Hours"] + df_clean["Social_Media_Hours"]

    return df_clean


@st.cache_data
def get_test_split():
    """
    Rebuilds the SAME held-out test set the notebook evaluated on
    (same random_state, same split ratio) so the Final Evaluation page
    can show a genuine confusion matrix / ROC curve from the already
    -trained Logistic Regression model. No fitting happens here -
    the scaler and model are only used with .transform() / .predict().
    """
    df_clean = clean_and_engineer(raw_df)
    X = df_clean.drop(columns="Depression")[FEATURE_NAMES]
    y = df_clean["Depression"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_test_scaled = X_test.copy()
    X_test_scaled[NUMERICAL_COLUMNS] = scaler.transform(X_test[NUMERICAL_COLUMNS])
    return X_test_scaled, y_test


def build_feature_row(age, gender, cgpa, sleep, study, social, activity,
                       stress, department):
    """Turns raw user inputs into a properly ordered, scaled feature row."""
    row = {name: 0 for name in FEATURE_NAMES}

    row["Age"] = age
    row["Gender"] = 1 if gender == "Female" else 0
    row["CGPA"] = cgpa
    row["Sleep_Duration"] = sleep
    row["Study_Hours"] = study
    row["Social_Media_Hours"] = social
    row["Physical_Activity"] = activity
    row["Stress_Level"] = stress

    # one-hot encoding (Arts is the dropped reference category)
    dummy_col = f"Department_{department}"
    if dummy_col in row:
        row[dummy_col] = 1

    # engineered features
    row["Sleep_Deficit"] = max(0, 8 - sleep)
    row["Total_Daily_Load"] = study + social

    X_input = pd.DataFrame([row])[FEATURE_NAMES]
    X_input[NUMERICAL_COLUMNS] = scaler.transform(X_input[NUMERICAL_COLUMNS])
    return X_input


# ----------------------------------------------------------------------
# 4. STATIC RESULTS FROM THE NOTEBOOK  (Model Comparison page)
#    These numbers were copied directly from the notebook's own output
#    cells - nothing is recomputed, since re-training is not wanted here.
# ----------------------------------------------------------------------
DEFAULT_RESULTS = pd.DataFrame({
    "Model": [
        "Logistic Regression", "Decision Tree", "Random Forest", "KNN", "SVM",
        "Logistic Regression", "Decision Tree", "Random Forest", "KNN", "SVM",
    ],
    "Setup": [
        "With PCA", "With PCA", "With PCA", "With PCA", "With PCA",
        "Without PCA", "Without PCA", "Without PCA", "Without PCA", "Without PCA",
    ],
    "Accuracy": [0.8994, 0.8216, 0.8994, 0.8892, 0.8994,
                 0.8994, 0.8162, 0.8992, 0.8914, 0.8994],
    "Precision": [0.0000, 0.1689, 0.5062, 0.2437, 0.0000,
                  0.0000, 0.1510, 0.4824, 0.2846, 0.0000],
    "Recall": [0.0000, 0.1973, 0.0204, 0.0482, 0.0000,
               0.0000, 0.1789, 0.0204, 0.0522, 0.0000],
    "F1-score": [0.0000, 0.1820, 0.0392, 0.0805, 0.0000,
                 0.0000, 0.1638, 0.0391, 0.0882, 0.0000],
})

IMPROVED_RESULTS = pd.DataFrame({
    "Model": ["Logistic Regression (tuned, balanced)", "Decision Tree (tuned)"],
    "Best Parameters": [
        "C=1, solver='lbfgs', class_weight='balanced'",
        "criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1",
    ],
    "Accuracy": [0.6190, 0.8162],
    "Precision": [0.1622, 0.1510],
    "Recall": [0.6695, 0.1789],
    "F1-score": [0.2612, 0.1638],
})

FINAL_METRICS = {
    "Accuracy": 0.6190,
    "Precision": 0.1622,
    "Recall": 0.6695,
    "F1-score": 0.2612,
    "ROC-AUC": 0.6796,
}

# ----------------------------------------------------------------------
# 5. SIDEBAR NAVIGATION
# ----------------------------------------------------------------------
st.sidebar.markdown("## 🧠 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📊 EDA", "⚖️ Model Comparison", "🔮 Prediction", "✅ Final Evaluation"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<span class='footnote' style='color:#EAF6F3;'>"
    "Built with saved models from Atheer's notebook.<br>"
    "No model is re-trained inside this app."
    "</span>",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# 6. HOME PAGE
# ----------------------------------------------------------------------
if page == "🏠 Home":
    st.markdown(
        """
        <div class="hero">
            <h1>🧠 Student Depression Risk Predictor</h1>
            <p>An end-to-end machine learning project exploring how demographic, academic
            and lifestyle factors relate to depression risk in students.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(f"<div class='metric-card'><h3>{raw_df.shape[0]:,}</h3>Students</div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-card'><h3>{raw_df.shape[1] - 1}</h3>Features</div>", unsafe_allow_html=True)
    dep_pct = raw_df["Depression"].mean() * 100
    col3.markdown(f"<div class='metric-card'><h3>{dep_pct:.1f}%</h3>Depression rate</div>", unsafe_allow_html=True)
    col4.markdown("<div class='metric-card'><h3>2</h3>Final models</div>", unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([1.3, 1])
    with left:
        st.subheader("About this project")
        st.write(
            """
            This app is built on top of a complete machine learning notebook that:
            - Explored and cleaned a **100,000-student** lifestyle & mental health dataset
            - Engineered two new features: **Sleep Deficit** and **Total Daily Load**
            - Compared **5 algorithms** (Logistic Regression, Decision Tree, Random Forest,
              KNN, SVM), with and without PCA
            - Tuned the two most promising models with **GridSearchCV**
            - Selected **Logistic Regression** as the final model because, on this
              imbalanced dataset, **Recall** (catching students who are actually at risk)
              matters more than raw accuracy
            """
        )
        st.subheader("How to use this app")
        st.write(
            """
            Use the sidebar to move between sections:
            - **📊 EDA** — explore the raw dataset visually
            - **⚖️ Model Comparison** — see how all 5 models performed
            - **🔮 Prediction** — enter a student's information and get a live prediction
            - **✅ Final Evaluation** — metrics, confusion matrix & ROC curve of the final model
            """
        )
    with right:
        st.subheader("Target balance")
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
        counts = raw_df["Depression"].value_counts()
        colors = [PRIMARY, WARN]
        ax.pie(
            counts, labels=["Not Depressed", "Depressed"], autopct="%1.1f%%",
            colors=colors, startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2},
        )
        ax.set_title("Depression class distribution")
        st.pyplot(fig)

    st.info(
        "⚠️ **Disclaimer:** This tool is an academic machine learning project, "
        "not a clinical or diagnostic instrument. It should never replace advice "
        "from a qualified mental health professional.",
        icon="⚠️",
    )

# ----------------------------------------------------------------------
# 7. EDA PAGE
# ----------------------------------------------------------------------
elif page == "📊 EDA":
    st.header("📊 Exploratory Data Analysis")
    st.caption("All charts below are generated live from the original dataset.")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Overview", "Numerical Features", "Categorical Features", "Relationships"]
    )

    # --- Overview ---
    with tab1:
        st.subheader("Dataset snapshot")
        st.dataframe(raw_df.head(10), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.write("**Summary statistics (numerical)**")
            st.dataframe(raw_df.describe().round(2), use_container_width=True)
        with c2:
            st.write("**Missing values**")
            missing = raw_df.isnull().sum()
            st.dataframe(missing[missing >= 0].rename("Missing Count"), use_container_width=True)
            st.write(f"Duplicate rows: **{raw_df.duplicated().sum()}**")

        st.write("**Target variable: Depression**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.countplot(data=raw_df, x="Depression", ax=ax, palette=[PRIMARY, WARN])
        ax.set_title("Distribution of Depression")
        st.pyplot(fig)
        st.caption(
            f"The target is imbalanced: about **{(1 - raw_df['Depression'].mean()) * 100:.1f}%** "
            f"of students are not depressed vs **{raw_df['Depression'].mean() * 100:.1f}%** who are — "
            "this is why Recall/F1 matter more than Accuracy for this problem."
        )

    # --- Numerical features ---
    with tab2:
        numerical_features = ["Age", "CGPA", "Sleep_Duration", "Study_Hours",
                               "Social_Media_Hours", "Physical_Activity", "Stress_Level"]
        st.subheader("Distributions")
        fig, axes = plt.subplots(3, 3, figsize=(13, 9))
        axes = axes.flatten()
        for i, col in enumerate(numerical_features):
            sns.histplot(raw_df[col], bins=20, ax=axes[i], color=PRIMARY, edgecolor="white")
            axes[i].set_title(col)
        for j in range(len(numerical_features), len(axes)):
            axes[j].axis("off")
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Feature vs Depression")
        chosen_feature = st.selectbox("Choose a numerical feature", numerical_features)
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        sns.boxplot(data=raw_df, x="Depression", y=chosen_feature, ax=ax2,
                    palette=[PRIMARY, WARN])
        ax2.set_title(f"{chosen_feature} by Depression status")
        st.pyplot(fig2)

    # --- Categorical features ---
    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Gender distribution**")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.countplot(data=raw_df, x="Gender", ax=ax, palette=[PRIMARY, ACCENT])
            st.pyplot(fig)
        with c2:
            st.write("**Department distribution**")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.countplot(data=raw_df, x="Department", ax=ax, color=PRIMARY)
            plt.xticks(rotation=20)
            st.pyplot(fig)

        st.write("**Depression rate by Department**")
        rate_by_dept = raw_df.groupby("Department")["Depression"].mean().sort_values(ascending=False) * 100
        fig, ax = plt.subplots(figsize=(7, 3.5))
        sns.barplot(x=rate_by_dept.index, y=rate_by_dept.values, ax=ax, color=PRIMARY)
        ax.set_ylabel("Depression rate (%)")
        plt.xticks(rotation=20)
        st.pyplot(fig)

    # --- Relationships / correlation ---
    with tab4:
        st.subheader("Correlation heatmap")
        corr_df = raw_df.drop(columns="Student_ID").copy()
        corr_df["Depression"] = corr_df["Depression"].astype(int)
        corr_df["Gender"] = corr_df["Gender"].map({"Male": 0, "Female": 1})
        corr_matrix = corr_df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="crest", linewidths=0.5, ax=ax)
        st.pyplot(fig)
        st.caption(
            "Most features show only a weak individual relationship with Depression, "
            "but Physical Activity, Sleep Duration and Stress Level are noticeably "
            "correlated with each other."
        )

# ----------------------------------------------------------------------
# 8. MODEL COMPARISON PAGE  (display only - no training happens here)
# ----------------------------------------------------------------------
elif page == "⚖️ Model Comparison":
    st.header("⚖️ Model Comparison")
    st.caption("Results copied directly from the notebook's experiments. No models are trained in this app.")

    st.subheader("Step 1 — Five models with default parameters")
    st.dataframe(
        DEFAULT_RESULTS.style.background_gradient(subset=["Recall", "F1-score"], cmap="Greens"),
        use_container_width=True,
    )
    st.caption(
        "Logistic Regression, Random Forest and SVM reached the highest **accuracy** "
        "(~90%) simply by predicting 'not depressed' almost every time — their Recall "
        "is near 0. The Decision Tree had the best Recall/F1 among the default models."
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    plot_df = DEFAULT_RESULTS[DEFAULT_RESULTS["Setup"] == "Without PCA"]
    x = np.arange(len(plot_df))
    width = 0.35
    ax.bar(x - width / 2, plot_df["Recall"], width, label="Recall", color=PRIMARY)
    ax.bar(x + width / 2, plot_df["F1-score"], width, label="F1-score", color=ACCENT)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["Model"], rotation=15)
    ax.set_title("Recall & F1-score by model (without PCA)")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Step 2 — Tuning with GridSearchCV (class_weight='balanced')")
    st.write(
        "Because the dataset is imbalanced, Logistic Regression and Decision Tree were "
        "improved using `class_weight='balanced'` and `GridSearchCV` (optimizing for Recall)."
    )
    st.dataframe(IMPROVED_RESULTS, use_container_width=True)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    metrics_plot = IMPROVED_RESULTS.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-score"]]
    metrics_plot.T.plot(kind="bar", ax=ax2, color=[PRIMARY, ACCENT])
    ax2.set_ylabel("Score")
    ax2.set_title("Tuned models comparison")
    plt.xticks(rotation=0)
    st.pyplot(fig2)

    st.success(
        "**Final choice: Logistic Regression.** The Decision Tree reached higher accuracy "
        "(81.6%), but Logistic Regression achieved a much higher **Recall (66.9%)** — "
        "meaning it catches far more of the students who are actually at risk of "
        "depression, which is the priority for this problem."
    )

# ----------------------------------------------------------------------
# 9. PREDICTION PAGE
# ----------------------------------------------------------------------
elif page == "🔮 Prediction":
    st.header("🔮 Predict Depression Risk")
    st.caption("Enter a student's information below and choose which trained model to use.")

    model_choice = st.radio(
        "Choose a model", ["Logistic Regression (final model)", "Decision Tree"], horizontal=True
    )

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Age", 18, 24, 21)
            gender = st.selectbox("Gender", ["Male", "Female"])
            cgpa = st.slider("CGPA", 1.5, 4.0, 2.9, step=0.01)
        with c2:
            sleep = st.slider("Sleep Duration (hours/night)", 3.0, 10.0, 7.0, step=0.1)
            study = st.slider("Study Hours (per day)", 0.0, 10.0, 4.5, step=0.1)
            social = st.slider("Social Media Hours (per day)", 0.0, 8.0, 3.5, step=0.1)
        with c3:
            activity = st.slider("Physical Activity (minutes/week)", 0, 150, 74)
            stress = st.slider("Stress Level (1 = low, 10 = high)", 1, 10, 4)
            department = st.selectbox("Department", DEPARTMENTS)

        submitted = st.form_submit_button("Predict", use_container_width=True)

    if submitted:
        X_input = build_feature_row(age, gender, cgpa, sleep, study, social,
                                     activity, stress, department)

        model = log_model if model_choice.startswith("Logistic") else tree_model
        pred = model.predict(X_input)[0]
        prob = model.predict_proba(X_input)[0][1]

        st.write("")
        col1, col2 = st.columns([1, 1.3])
        with col1:
            if pred == 1:
                st.markdown(
                    f"<div class='badge-high'>⚠️ Higher risk of depression<br>"
                    f"Estimated probability: {prob:.1%}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='badge-low'>✅ Lower risk of depression<br>"
                    f"Estimated probability: {prob:.1%}</div>",
                    unsafe_allow_html=True,
                )
        with col2:
            fig, ax = plt.subplots(figsize=(5, 0.9))
            ax.barh([0], [1], color="#E9ECEF")
            ax.barh([0], [prob], color=WARN if pred == 1 else PRIMARY)
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.set_xlabel("Predicted probability of depression")
            st.pyplot(fig)

        st.caption(
            f"Model used: **{model_choice}** — "
            "this reflects the pattern the model learned in training and is not a "
            "clinical diagnosis."
        )
        st.info(
            "💙 If you or someone you know is struggling with their mental health, "
            "please reach out to a counselor, doctor, or a trusted support line. "
            "This app is for educational purposes only.",
            icon="💙",
        )

# ----------------------------------------------------------------------
# 10. FINAL EVALUATION PAGE
# ----------------------------------------------------------------------
elif page == "✅ Final Evaluation":
    st.header("✅ Final Model Evaluation")
    st.caption(
        "Final model: **Logistic Regression (tuned, class_weight='balanced')**. "
        "Metrics are computed on the same held-out test split used in the notebook, "
        "using the already-trained model only — no fitting happens here."
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{FINAL_METRICS['Accuracy']:.1%}")
    m2.metric("Precision", f"{FINAL_METRICS['Precision']:.1%}")
    m3.metric("Recall", f"{FINAL_METRICS['Recall']:.1%}")
    m4.metric("F1-score", f"{FINAL_METRICS['F1-score']:.1%}")
    m5.metric("ROC-AUC", f"{FINAL_METRICS['ROC-AUC']:.1%}")

    X_test_scaled, y_test = get_test_split()
    y_pred = log_model.predict(X_test_scaled)
    y_prob = log_model.predict_proba(X_test_scaled)[:, 1]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4.5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                       display_labels=["Not Depressed", "Depressed"])
        disp.plot(ax=ax, cmap="BuGn", colorbar=False)
        st.pyplot(fig)

    with col2:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.plot(fpr, tpr, color=PRIMARY, linewidth=2,
                label=f"AUC = {roc_auc_score(y_test, y_prob):.3f}")
        ax.plot([0, 1], [0, 1], "--", color="#AAAAAA")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    st.subheader("Conclusion")
    st.write(
        """
        Five models were compared with default parameters, both with and without PCA.
        Most models reached ~90% accuracy purely by predicting "not depressed" almost
        every time, which is not useful for identifying at-risk students. After tuning
        with `class_weight='balanced'` and `GridSearchCV`, **Logistic Regression** was
        chosen as the final model because it achieves the highest **Recall (66.9%)** —
        correctly flagging two out of three students who are actually at risk — at the
        cost of some accuracy and precision. On an imbalanced mental-health dataset like
        this one, minimizing missed at-risk students is more important than overall
        accuracy.
        """
    )
    st.caption(
        "Future improvements could include collecting more data, engineering additional "
        "lifestyle features, or exploring cost-sensitive / ensemble methods."
    )
