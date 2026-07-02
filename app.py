"""
Student Lifestyle & Depression Explorer
----------------------------------------
Streamlit app for a beginner AI & Machine Learning course project.

IMPORTANT:
This app does NOT train any model and does NOT need any dataset upload.
All EDA, cleaning, feature engineering, PCA, model training, tuning, and
evaluation were already done in the project notebook. This app only:
    1) shows the charts/figures already exported (saved as images) from the notebook
    2) loads the saved model files (.pkl) with joblib.load()
    3) lets the user enter values with sliders
    4) makes a prediction with the model the user picks
    5) displays the prediction, probability, and a simple explanation

Required model files (must sit in the same folder as this app.py):
    Logistic_model.pkl        -> tuned, class-balanced Logistic Regression
    decision_tree_model.pkl   -> tuned Decision Tree
    scaler.pkl                -> the StandardScaler fit in the notebook
    feature_names.pkl         -> list of feature column names, in order

Required chart images (must sit in an "images" folder next to app.py):
    depression_distribution.png
    numerical_features_histograms.png
    gender_distribution.png
    department_distribution.png
    correlation_heatmap.png
    boxplot_age.png
    boxplot_cgpa.png
    boxplot_sleep_duration.png
    boxplot_study_hours.png
    boxplot_social_media_hours.png
    boxplot_physical_activity.png
    boxplot_stress_level.png
    confusion_matrix.png
    roc_curve.png
(Export each one from your notebook with plt.savefig("images/<name>.png"))
"""

import os
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------------------------------------------------
# PAGE CONFIG + THEME (unchanged from the original design)
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Student Lifestyle & Wellbeing Explorer",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = {
    "indigo": "#6C63FF",
    "teal": "#4ECDC4",
    "coral": "#FFB4A2",
    "amber": "#FFD07B",
    "ink": "#2D2A4A",
    "bg": "#F8F6FC",
    "card": "#FFFFFF",
    "muted": "#8B87A8",
}
SEQ_PALETTE = ["#6C63FF", "#4ECDC4", "#FFB4A2", "#FFD07B", "#A29BFE", "#55E6C1"]

st.markdown(f"""
<style>
    .stApp {{
        background-color: {PALETTE['bg']};
    }}
    h1, h2, h3 {{
        color: {PALETTE['ink']};
        font-family: 'Trebuchet MS', sans-serif;
    }}
    div[data-testid="stMetric"] {{
        background: {PALETTE['card']};
        border-radius: 16px;
        padding: 16px 12px;
        box-shadow: 0 2px 10px rgba(108,99,255,0.08);
        border: 1px solid rgba(108,99,255,0.10);
    }}
    div[data-testid="stMetricValue"] {{
        color: {PALETTE['indigo']};
    }}
    section[data-testid="stSidebar"] {{
        background-color: #EFEBFB;
    }}
    .insight-card {{
        background: {PALETTE['card']};
        border-left: 5px solid {PALETTE['teal']};
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(45,42,74,0.06);
    }}
    .banner {{
        background: linear-gradient(135deg, #6C63FF 0%, #4ECDC4 100%);
        border-radius: 18px;
        padding: 28px 32px;
        color: white;
        margin-bottom: 22px;
    }}
    .pill {{
        display:inline-block; padding:4px 14px; border-radius:999px;
        background:#EFEBFB; color:{PALETTE['indigo']}; font-size:0.8rem; margin-right:6px;
    }}
</style>
""", unsafe_allow_html=True)


def plotly_theme(fig, height=380):
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE["ink"], family="Trebuchet MS"),
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ----------------------------------------------------------------------
# Helper to show a saved notebook chart image (no data needed here at all)
# ----------------------------------------------------------------------
IMAGES_DIR = "images"


def show_image(filename, caption=None):
    path = os.path.join(IMAGES_DIR, filename)
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.info(
            f"📁 Chart not found yet. Export this figure from your notebook and save it as "
            f"`{IMAGES_DIR}/{filename}` (e.g. `plt.savefig('{IMAGES_DIR}/{filename}')`)."
        )


# ----------------------------------------------------------------------
# CONSTANTS — real numbers copied from the notebook's own printed output
# (not uploaded data, just the summary statistics you already calculated)
# ----------------------------------------------------------------------
TOTAL_STUDENTS = 100000
DEPRESSION_RATE = 10.06     # %
AVG_SLEEP = 7.0              # hours
AVG_STRESS = 4.13            # out of 10

NUMERICAL_FEATURES = [
    "Age", "CGPA", "Sleep_Duration", "Study_Hours",
    "Social_Media_Hours", "Physical_Activity", "Stress_Level"
]

# These are exactly the columns the notebook scaled with StandardScaler
# (see the "Preprocessing" section of the notebook).
NUMERICAL_COLUMNS_TO_SCALE = [
    "Age", "CGPA", "Sleep_Duration", "Study_Hours", "Social_Media_Hours",
    "Physical_Activity", "Stress_Level", "Sleep_Deficit", "Total_Daily_Load"
]

DEPARTMENTS = ["Science", "Engineering", "Medical", "Arts", "Business"]
GENDERS = ["Male", "Female"]

# Results copied directly from the notebook's own printed output.
# Nothing here is retrained -- these are the numbers the notebook produced.
BENCHMARK_RESULTS = pd.DataFrame([
    # Default models, trained WITH PCA
    {"Model": "Logistic Regression", "Stage": "Default (with PCA)",    "Accuracy": 0.8994, "Precision": 0.0000, "Recall": 0.0000, "F1-score": 0.0000},
    {"Model": "Decision Tree",       "Stage": "Default (with PCA)",    "Accuracy": 0.8216, "Precision": 0.1689, "Recall": 0.1973, "F1-score": 0.1820},
    {"Model": "Random Forest",       "Stage": "Default (with PCA)",    "Accuracy": 0.8995, "Precision": 0.5062, "Recall": 0.0204, "F1-score": 0.0392},
    {"Model": "KNN",                 "Stage": "Default (with PCA)",    "Accuracy": 0.8892, "Precision": 0.2437, "Recall": 0.0482, "F1-score": 0.0805},
    {"Model": "SVM",                 "Stage": "Default (with PCA)",    "Accuracy": 0.8994, "Precision": 0.0000, "Recall": 0.0000, "F1-score": 0.0000},
    # Default models, trained WITHOUT PCA
    {"Model": "Logistic Regression", "Stage": "Default (without PCA)", "Accuracy": 0.8994, "Precision": 0.0000, "Recall": 0.0000, "F1-score": 0.0000},
    {"Model": "Decision Tree",       "Stage": "Default (without PCA)", "Accuracy": 0.8162, "Precision": 0.1510, "Recall": 0.1789, "F1-score": 0.1638},
    {"Model": "Random Forest",       "Stage": "Default (without PCA)", "Accuracy": 0.8993, "Precision": 0.4824, "Recall": 0.0204, "F1-score": 0.0391},
    {"Model": "KNN",                 "Stage": "Default (without PCA)", "Accuracy": 0.8915, "Precision": 0.2846, "Recall": 0.0522, "F1-score": 0.0882},
    {"Model": "SVM",                 "Stage": "Default (without PCA)", "Accuracy": 0.8994, "Precision": 0.0000, "Recall": 0.0000, "F1-score": 0.0000},
    # Final tuned models (GridSearchCV), no PCA
    {"Model": "Logistic Regression (Improved)", "Stage": "Tuned (final)", "Accuracy": 0.6190, "Precision": 0.1622, "Recall": 0.6695, "F1-score": 0.2612},
    {"Model": "Decision Tree (Improved)",       "Stage": "Tuned (final)", "Accuracy": 0.8162, "Precision": 0.1510, "Recall": 0.1789, "F1-score": 0.1638},
])
FINAL_ROC_AUC = 0.6796


# ----------------------------------------------------------------------
# LOAD THE SAVED MODEL FILES (no training happens here, only loading)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading saved models...")
def load_saved_models():
    logistic_model = joblib.load("Logistic_model.pkl")
    decision_tree_model = joblib.load("decision_tree_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return logistic_model, decision_tree_model, scaler, feature_names


models_loaded = True
try:
    logistic_model, decision_tree_model, scaler, feature_names = load_saved_models()
except FileNotFoundError as e:
    models_loaded = False
    missing_file_error = e


# ========================================================================
# SIDEBAR — navigation only (no dataset upload)
# ========================================================================
st.sidebar.markdown("## 🌿 Navigate")
page = st.sidebar.radio(
    "",
    ["🏠 Overview", "🔍 Explore the Data", "💡 Insights & Correlations",
     "🤖 Model Performance", "🧮 Risk Estimator"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<span class='pill'>Beginner AI/ML Project</span><span class='pill'>Streamlit</span>",
    unsafe_allow_html=True,
)

if not models_loaded:
    st.error(
        "⚠️ Could not find your saved model files. Please make sure these 4 files "
        "are in the **same folder** as `app.py`:\n\n"
        "- `Logistic_model.pkl`\n- `decision_tree_model.pkl`\n- `scaler.pkl`\n- `feature_names.pkl`\n\n"
        f"Technical detail: {missing_file_error}"
    )

# ========================================================================
# PAGE 1 — OVERVIEW
# ========================================================================
if page == "🏠 Overview":
    st.markdown("""
    <div class="banner">
        <h1 style="color:white; margin-bottom:4px;">🌿 Student Lifestyle & Depression Explorer</h1>
        <p style="font-size:1.05rem; opacity:0.95;">
        An interactive presentation of the EDA and machine-learning project on the
        <b>Student Depression & Lifestyle</b> dataset — exploring how sleep, stress,
        study habits, and screen time relate to student wellbeing.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students in dataset", f"{TOTAL_STUDENTS:,}")
    c2.metric("At-risk of Depression", f"{DEPRESSION_RATE:.2f}%")
    c3.metric("Avg. Sleep", f"{AVG_SLEEP:.1f} hrs")
    c4.metric("Avg. Stress Level", f"{AVG_STRESS:.2f} / 10")

    st.markdown("### 📋 About this project")
    colA, colB = st.columns([1.3, 1])
    with colA:
        st.markdown("""
        This app presents the findings of a student machine-learning project that set out to
        answer: **can lifestyle and academic factors help flag students who may be at risk of
        depression?**

        The pipeline covered (all done in the notebook, not in this app):
        - Cleaning and outlier handling (IQR capping) on sleep, study, social-media, and stress data
        - Feature engineering (**Sleep Deficit**, **Total Daily Load**)
        - Dimensionality reduction with PCA (14 features → 9 components, ~95% variance kept)
        - Five classifiers compared, then tuned with `GridSearchCV`
        - A final **class-balanced Logistic Regression**, chosen for its ability to catch more
          at-risk students (Recall), not just for raw accuracy
        """)
    with colB:
        show_image("depression_distribution.png", caption="Depression class balance (from the notebook)")

    st.markdown("""
    <div class="insight-card">
    ⚠️ <b>A gentle note:</b> this dataset and model are simplified educational tools, not
    diagnostic instruments. If you or someone you know is struggling, please reach out to a
    counselor, doctor, or a trusted support line — real support matters more than any model.
    </div>
    """, unsafe_allow_html=True)

# ========================================================================
# PAGE 2 — EXPLORE THE DATA (saved notebook charts only, no data needed)
# ========================================================================
elif page == "🔍 Explore the Data":
    st.markdown("## 🔍 Explore the Data")
    st.write("These charts were generated in the notebook and are simply displayed here.")

    tab1, tab2 = st.tabs(["📊 Numerical Features", "🧑‍🤝‍🧑 Demographics"])

    with tab1:
        show_image("numerical_features_histograms.png", caption="Distribution of all numerical features")

    with tab2:
        colL, colR = st.columns(2)
        with colL:
            show_image("gender_distribution.png", caption="Gender distribution")
        with colR:
            show_image("department_distribution.png", caption="Department distribution")

# ========================================================================
# PAGE 3 — INSIGHTS & CORRELATIONS (saved notebook charts only)
# ========================================================================
elif page == "💡 Insights & Correlations":
    st.markdown("## 💡 Insights & Correlations")

    show_image("correlation_heatmap.png", caption="Correlation heatmap of numerical features and Depression")

    st.markdown("### 🧠 Key findings from the analysis")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="insight-card">
        😴 <b>Sleep & stress move together.</b> Students who sleep closer to 7–8 hours tend to
        report noticeably lower stress levels than those getting much less sleep.
        </div>
        <div class="insight-card">
        🏃 <b>Physical activity helps too.</b> More active students also trend toward lower
        stress — the strongest relationships in the whole dataset are Sleep↔Stress and
        Physical Activity↔Stress.
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="insight-card">
        📉 <b>No single "smoking gun".</b> Individually, most lifestyle features only weakly
        correlate with Depression — risk looks like it comes from a <i>combination</i> of
        factors rather than one dominant cause.
        </div>
        <div class="insight-card">
        ⚖️ <b>The classes are imbalanced.</b> Only ~10% of students are labeled at-risk, which
        is exactly why accuracy alone is misleading and Recall/Precision/F1 matter more.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📦 Feature spread by Depression status")
    feat = st.selectbox("Pick a feature to view its boxplot", NUMERICAL_FEATURES)
    show_image(f"boxplot_{feat.lower()}.png", caption=f"{feat.replace('_', ' ')} by Depression status")

# ========================================================================
# PAGE 4 — MODEL PERFORMANCE (results + saved figures only, no retraining)
# ========================================================================
elif page == "🤖 Model Performance":
    st.markdown("## 🤖 Model Performance")
    st.write(
        "These are the exact results already produced in the notebook. Five classifiers "
        "were compared with and without PCA, and the two most promising were then tuned "
        "with `GridSearchCV`. Nothing is retrained on this page — the numbers and figures "
        "below are simply displayed."
    )

    stage_choice = st.multiselect(
        "Filter by stage", BENCHMARK_RESULTS["Stage"].unique().tolist(),
        default=BENCHMARK_RESULTS["Stage"].unique().tolist()
    )
    metric_choice = st.radio("Metric", ["Accuracy", "Precision", "Recall", "F1-score"], horizontal=True)

    view = BENCHMARK_RESULTS[BENCHMARK_RESULTS["Stage"].isin(stage_choice)]
    fig = px.bar(
        view.sort_values(metric_choice), x=metric_choice, y="Model",
        orientation="h", color="Stage", color_discrete_sequence=SEQ_PALETTE,
        title=f"Model comparison — {metric_choice}", barmode="group"
    )
    st.plotly_chart(plotly_theme(fig, 460), use_container_width=True)

    with st.expander("See the full results table"):
        st.dataframe(BENCHMARK_RESULTS, use_container_width=True)

    st.markdown("### 📈 Evaluation figures (final model)")
    colL, colR = st.columns(2)
    with colL:
        show_image("confusion_matrix.png", caption="Confusion Matrix — Improved Logistic Regression")
    with colR:
        show_image("roc_curve.png", caption=f"ROC Curve — AUC = {FINAL_ROC_AUC:.3f}")

    st.markdown(f"""
    <div class="insight-card">
    🎯 <b>Why Logistic Regression (Improved) was chosen, despite lower accuracy:</b> Random
    Forest and SVM reached ~90% accuracy mostly by predicting "no depression" almost every
    time — great accuracy, but they missed nearly every at-risk student (Recall near 0%).
    The tuned, class-balanced Logistic Regression trades some accuracy for a much higher
    <b>Recall (66.9%)</b>, catching far more of the students who actually need support.
    </div>
    """, unsafe_allow_html=True)

# ========================================================================
# PAGE 5 — RISK ESTIMATOR (uses the real saved models, no training here)
# ========================================================================
elif page == "🧮 Risk Estimator":
    st.markdown("## 🧮 Risk Estimator")
    st.markdown("""
    <div class="insight-card">
    🎓 <b>Educational demo only.</b> This tool uses the model you actually trained and saved
    in the notebook. It is <b>not</b> a diagnostic or clinical tool. If you have concerns
    about depression, please talk to a doctor, counselor, or a trusted person.
    </div>
    """, unsafe_allow_html=True)

    if not models_loaded:
        st.error("The saved model files are missing, so predictions can't be made. See the message at the top of the app.")
    else:
        st.markdown("### 1. Choose a model")
        model_choice = st.radio(
            "Which trained model should make the prediction?",
            ["Logistic Regression (Improved)", "Decision Tree (Improved)"],
            horizontal=True,
        )

        st.markdown("### 2. Adjust the sliders")
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Age", 18, 24, 21)
            gender = st.radio("Gender", GENDERS, horizontal=True)
            department = st.selectbox("Department", DEPARTMENTS)
        with c2:
            cgpa = st.slider("CGPA", 1.5, 4.0, 2.9, 0.05)
            sleep = st.slider("Sleep Duration (hrs)", 3.0, 12.0, 7.0, 0.5)
            study = st.slider("Study Hours / day", 0.0, 12.0, 4.5, 0.5)
        with c3:
            social = st.slider("Social Media Hours / day", 0.0, 10.0, 3.5, 0.5)
            activity = st.slider("Physical Activity (min/day)", 0, 149, 74)
            stress = st.slider("Stress Level (1–10)", 1, 10, 4)

        sleep_deficit = max(8 - sleep, 0)
        total_load = study + social

        # -----------------------------------------------------------
        # Build ONE row of input data, matching feature_names.pkl.
        # We go through the list of feature names the notebook saved,
        # and fill in the right value for each one. Any Department_*
        # column is set to 1 only if it matches the chosen department.
        # -----------------------------------------------------------
        input_row = {}
        for name in feature_names:
            if name == "Age":
                input_row[name] = age
            elif name == "Gender":
                input_row[name] = 1 if gender == "Female" else 0
            elif name == "CGPA":
                input_row[name] = cgpa
            elif name == "Sleep_Duration":
                input_row[name] = sleep
            elif name == "Study_Hours":
                input_row[name] = study
            elif name == "Social_Media_Hours":
                input_row[name] = social
            elif name == "Physical_Activity":
                input_row[name] = activity
            elif name == "Stress_Level":
                input_row[name] = stress
            elif name == "Sleep_Deficit":
                input_row[name] = sleep_deficit
            elif name == "Total_Daily_Load":
                input_row[name] = total_load
            elif name.startswith("Department_"):
                input_row[name] = 1 if name == f"Department_{department}" else 0
            else:
                # Any feature we don't recognize is set to 0 as a safe default.
                input_row[name] = 0

        # Put the values in a DataFrame with columns in the SAME ORDER as feature_names.pkl
        input_df = pd.DataFrame([input_row])[feature_names]

        # Scale only the numerical columns, using the notebook's saved scaler
        cols_to_scale = [c for c in NUMERICAL_COLUMNS_TO_SCALE if c in input_df.columns]
        input_scaled = input_df.copy()
        input_scaled[cols_to_scale] = scaler.transform(input_df[cols_to_scale])

        # Pick the model the user selected
        if model_choice == "Logistic Regression (Improved)":
            chosen_model = logistic_model
        else:
            chosen_model = decision_tree_model

        # Make the prediction with the real saved model
        prediction = chosen_model.predict(input_scaled)[0]
        probability = chosen_model.predict_proba(input_scaled)[0][1]

        st.markdown("### 3. Result")
        colL, colR = st.columns([1, 1.4])
        with colL:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": PALETTE["indigo"]},
                    "steps": [
                        {"range": [0, 33], "color": "#D6F5F0"},
                        {"range": [33, 66], "color": "#FFE9C7"},
                        {"range": [66, 100], "color": "#FFD6CC"},
                    ],
                },
                title={"text": "Predicted probability"},
            ))
            st.plotly_chart(plotly_theme(fig, 300), use_container_width=True)

        with colR:
            label = "Depressed" if prediction == 1 else "Not Depressed"
            st.metric("Prediction", label)
            st.metric("Probability of Depression", f"{probability*100:.1f}%")

            st.markdown("**Simple explanation:**")
            reasons = []
            if stress >= 7:
                reasons.append("a high reported **stress level**")
            if sleep_deficit >= 2:
                reasons.append("a noticeable **sleep deficit** (getting much less than 8 hours)")
            if total_load >= 10:
                reasons.append("a high **combined study + social media load**")
            if activity <= 30:
                reasons.append("**low physical activity**")

            if prediction == 1:
                if reasons:
                    st.write(
                        "The model predicted a higher depression risk, which lines up with "
                        + ", ".join(reasons) + " in the values you entered."
                    )
                else:
                    st.write(
                        "The model predicted a higher depression risk based on the overall "
                        "combination of inputs, even though no single factor stands out strongly."
                    )
            else:
                st.write(
                    "The model predicted a lower depression risk. Based on the notebook's "
                    "findings, healthier sleep, lower stress, and a manageable daily load are "
                    "generally associated with lower risk."
                )

            st.caption(
                f"Model used: **{model_choice}**. Remember this model favors catching more "
                "at-risk students (higher Recall) over always being precise, so it may flag "
                "some students who are not actually at risk."
            )

st.markdown("---")
st.caption(
    "Built with Streamlit · Uses the trained models saved from the Student Depression & "
    "Lifestyle EDA + ML project notebook · For education only — not a substitute for "
    "professional mental health advice."
)
