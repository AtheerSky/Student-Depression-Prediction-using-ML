import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from data_utils import (
    generate_synthetic_data, train_demo_model,
    NUMERICAL_FEATURES, DEPARTMENTS, BENCHMARK_RESULTS, FINAL_ROC_AUC
)

# ----------------------------------------------------------------------
# PAGE CONFIG + THEME
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
DEPR_COLORS = {"No Depression": "#4ECDC4", "Depression": "#FF9F86"}

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
# SIDEBAR — navigation + data source
# ----------------------------------------------------------------------
st.sidebar.markdown("## 🌿 Navigate")
page = st.sidebar.radio(
    "",
    ["🏠 Overview", "🔍 Explore the Data", "💡 Insights & Correlations",
     "🤖 Model Performance", "🧮 Try the Risk Estimator"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 Data source")
uploaded = st.sidebar.file_uploader("Upload the real dataset CSV (optional)", type="csv")
st.sidebar.caption(
    "By default this app explores a **synthetic sample** generated to match "
    "the statistics reported in the EDA notebook (100k-row Kaggle dataset). "
    "Upload the real CSV to replace it everywhere."
)

if uploaded is not None:
    df = pd.read_csv(uploaded)
    if "Sleep_Deficit" not in df.columns and "Sleep_Duration" in df.columns:
        df["Sleep_Deficit"] = (8 - df["Sleep_Duration"]).clip(lower=0)
    if "Total_Daily_Load" not in df.columns and {"Study_Hours", "Social_Media_Hours"}.issubset(df.columns):
        df["Total_Daily_Load"] = df["Study_Hours"] + df["Social_Media_Hours"]
    data_source = "your uploaded file"
else:
    df = generate_synthetic_data()
    data_source = "a synthetic sample matching the notebook's reported statistics"

df["Depression_Label"] = df["Depression"].map({True: "Depression", False: "No Depression",
                                                 1: "Depression", 0: "No Depression"})

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<span class='pill'>Beginner EDA</span><span class='pill'>Streamlit</span>",
    unsafe_allow_html=True,
)

# ========================================================================
# PAGE 1 — OVERVIEW
# ========================================================================
if page == "🏠 Overview":
    st.markdown("""
    <div class="banner">
        <h1 style="color:white; margin-bottom:4px;">🌿 Student Lifestyle & Depression Explorer</h1>
        <p style="font-size:1.05rem; opacity:0.95;">
        An interactive walkthrough of the EDA and machine-learning project on the
        <b>Student Depression & Lifestyle</b> dataset — exploring how sleep, stress,
        study habits, and screen time relate to student wellbeing.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.caption(f"Currently exploring: **{data_source}** · {len(df):,} students")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students", f"{len(df):,}")
    c2.metric("At-risk of Depression", f"{df['Depression'].mean()*100:.1f}%")
    c3.metric("Avg. Sleep", f"{df['Sleep_Duration'].mean():.1f} hrs")
    c4.metric("Avg. Stress Level", f"{df['Stress_Level'].mean():.1f} / 10")

    st.markdown("### 📋 About this project")
    colA, colB = st.columns([1.3, 1])
    with colA:
        st.markdown("""
        This app presents the findings of a student machine-learning project that set out to
        answer: **can lifestyle and academic factors help flag students who may be at risk of
        depression?**

        The pipeline covered:
        - Cleaning and outlier handling (IQR capping) on sleep, study, social-media, and stress data
        - Feature engineering (**Sleep Deficit**, **Total Daily Load**)
        - Dimensionality reduction with PCA (14 features → 9 components, ~95% variance kept)
        - Five classifiers compared, then tuned with `GridSearchCV`
        - A final **class-balanced Logistic Regression**, chosen for its ability to catch more
          at-risk students (Recall), not just for raw accuracy
        """)
    with colB:
        dep_counts = df["Depression_Label"].value_counts().reset_index()
        dep_counts.columns = ["Status", "Count"]
        fig = px.pie(
            dep_counts, names="Status", values="Count", hole=0.55,
            color="Status", color_discrete_map=DEPR_COLORS,
            title="Depression Class Balance",
        )
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(plotly_theme(fig, 320), use_container_width=True)

    st.markdown("""
    <div class="insight-card">
    ⚠️ <b>A gentle note:</b> this dataset and model are simplified educational tools, not
    diagnostic instruments. If you or someone you know is struggling, please reach out to a
    counselor, doctor, or a trusted support line — real support matters more than any model.
    </div>
    """, unsafe_allow_html=True)

# ========================================================================
# PAGE 2 — EXPLORE THE DATA
# ========================================================================
elif page == "🔍 Explore the Data":
    st.markdown("## 🔍 Explore the Data")
    st.write("Filter the sample and see how distributions shift in real time.")

    f1, f2, f3 = st.columns(3)
    with f1:
        genders = st.multiselect("Gender", sorted(df["Gender"].unique()), default=list(df["Gender"].unique()))
    with f2:
        depts = st.multiselect("Department", sorted(df["Department"].unique()), default=list(df["Department"].unique()))
    with f3:
        age_range = st.slider("Age range", int(df["Age"].min()), int(df["Age"].max()),
                               (int(df["Age"].min()), int(df["Age"].max())))

    fdf = df[
        df["Gender"].isin(genders) & df["Department"].isin(depts)
        & df["Age"].between(age_range[0], age_range[1])
    ]
    st.caption(f"Showing **{len(fdf):,}** of {len(df):,} students matching your filters.")

    tab1, tab2, tab3 = st.tabs(["📊 Distributions", "🧑‍🤝‍🧑 Demographics", "🗂️ Raw data"])

    with tab1:
        feature = st.selectbox("Choose a numerical feature", NUMERICAL_FEATURES, index=2)
        colL, colR = st.columns(2)
        with colL:
            fig = px.histogram(
                fdf, x=feature, nbins=30, color_discrete_sequence=[PALETTE["indigo"]],
                title=f"Distribution of {feature.replace('_', ' ')}"
            )
            st.plotly_chart(plotly_theme(fig), use_container_width=True)
        with colR:
            fig2 = px.box(
                fdf, x="Depression_Label", y=feature, color="Depression_Label",
                color_discrete_map=DEPR_COLORS,
                title=f"{feature.replace('_', ' ')} by Depression status"
            )
            fig2.update_layout(showlegend=False)
            st.plotly_chart(plotly_theme(fig2), use_container_width=True)

    with tab2:
        colL, colR = st.columns(2)
        with colL:
            g = fdf["Gender"].value_counts().reset_index()
            g.columns = ["Gender", "Count"]
            fig = px.bar(g, x="Gender", y="Count", color="Gender",
                         color_discrete_sequence=SEQ_PALETTE, title="Gender distribution")
            fig.update_layout(showlegend=False)
            st.plotly_chart(plotly_theme(fig, 340), use_container_width=True)
        with colR:
            d = fdf["Department"].value_counts().reset_index()
            d.columns = ["Department", "Count"]
            fig = px.bar(d, x="Department", y="Count", color="Department",
                         color_discrete_sequence=SEQ_PALETTE, title="Department distribution")
            fig.update_layout(showlegend=False)
            st.plotly_chart(plotly_theme(fig, 340), use_container_width=True)

    with tab3:
        st.dataframe(fdf.drop(columns=["Depression_Label"]), use_container_width=True, height=380)

# ========================================================================
# PAGE 3 — INSIGHTS & CORRELATIONS
# ========================================================================
elif page == "💡 Insights & Correlations":
    st.markdown("## 💡 Insights & Correlations")

    corr_df = df.copy()
    corr_df["Depression"] = corr_df["Depression"].astype(int)
    num_cols = NUMERICAL_FEATURES + ["Sleep_Deficit", "Total_Daily_Load", "Depression"]
    num_cols = [c for c in num_cols if c in corr_df.columns]
    corr = corr_df[num_cols].corr()

    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale=["#4ECDC4", "#F8F6FC", "#FF9F86"],
        title="Correlation Heatmap", aspect="auto"
    )
    st.plotly_chart(plotly_theme(fig, 480), use_container_width=True)

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
    feat = st.selectbox("Pick a feature to compare", NUMERICAL_FEATURES, index=6, key="insight_feat")
    fig2 = px.violin(
        df, x="Depression_Label", y=feat, color="Depression_Label",
        color_discrete_map=DEPR_COLORS, box=True, points=False,
        title=f"{feat.replace('_', ' ')}: Depression vs. No Depression"
    )
    fig2.update_layout(showlegend=False)
    st.plotly_chart(plotly_theme(fig2), use_container_width=True)

# ========================================================================
# PAGE 4 — MODEL PERFORMANCE
# ========================================================================
elif page == "🤖 Model Performance":
    st.markdown("## 🤖 Model Performance")
    st.write(
        "Five classifiers were trained and compared, then the two most promising were tuned "
        "with `GridSearchCV`. The final row is the model that was selected."
    )

    metric_choice = st.radio("Metric", ["Accuracy", "Precision", "Recall", "F1-score"],
                              horizontal=True)
    fig = px.bar(
        BENCHMARK_RESULTS.sort_values(metric_choice), x=metric_choice, y="Model",
        orientation="h", color=metric_choice, color_continuous_scale=["#EFEBFB", "#6C63FF"],
        title=f"Model comparison — {metric_choice}"
    )
    st.plotly_chart(plotly_theme(fig, 420), use_container_width=True)

    st.markdown("""
    <div class="insight-card">
    🎯 <b>Why Logistic Regression won, despite lower accuracy:</b> Random Forest and SVM
    reached ~90% accuracy mostly by predicting "no depression" almost every time — great
    accuracy, but they missed nearly every at-risk student (Recall near 0). The tuned,
    class-balanced Logistic Regression trades some accuracy for a much higher <b>Recall
    (66.9%)</b>, catching far more of the students who actually need support.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🧪 Try it live")
    st.caption(
        "For hands-on exploration, the section below trains a small demo Logistic Regression "
        "on the current in-app dataset (same class-balanced approach as the notebook)."
    )

    model, scaler, feature_cols, num_cols, metrics = train_demo_model(df)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
    m2.metric("Precision", f"{metrics['precision']*100:.1f}%")
    m3.metric("Recall", f"{metrics['recall']*100:.1f}%")
    m4.metric("F1-score", f"{metrics['f1']*100:.1f}%")
    m5.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

    colL, colR = st.columns(2)
    with colL:
        cm = metrics["confusion_matrix"]
        fig_cm = px.imshow(
            cm, text_auto=True, color_continuous_scale=["#F8F6FC", "#6C63FF"],
            x=["Predicted: No", "Predicted: Yes"], y=["Actual: No", "Actual: Yes"],
            title="Confusion Matrix (demo model)"
        )
        st.plotly_chart(plotly_theme(fig_cm, 360), use_container_width=True)
    with colR:
        fpr, tpr, _ = metrics["roc_curve"]
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                      line=dict(color=PALETTE["indigo"], width=3),
                                      name=f"AUC = {metrics['roc_auc']:.3f}"))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                      line=dict(color=PALETTE["muted"], dash="dash"),
                                      name="Random guess"))
        fig_roc.update_layout(title="ROC Curve (demo model)",
                               xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(plotly_theme(fig_roc, 360), use_container_width=True)

    st.caption(
        f"Reference — the notebook's final tuned model reported ROC-AUC ≈ {FINAL_ROC_AUC:.3f} "
        "on the full 100k-row dataset; the demo model above will vary slightly since it is "
        "trained on this smaller in-app sample."
    )

# ========================================================================
# PAGE 5 — RISK ESTIMATOR
# ========================================================================
elif page == "🧮 Try the Risk Estimator":
    st.markdown("## 🧮 Try the Risk Estimator")
    st.markdown("""
    <div class="insight-card">
    🎓 <b>Educational demo only.</b> This slider tool runs a small logistic-regression model
    trained on in-app sample data. It is meant to illustrate how the project's model works —
    it is <b>not</b> a diagnostic or clinical tool. If you have concerns about depression,
    please talk to a doctor, counselor, or a trusted person.
    </div>
    """, unsafe_allow_html=True)

    model, scaler, feature_cols, num_cols, metrics = train_demo_model(df)

    st.markdown("### Adjust the sliders")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Age", 18, 24, 21)
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
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

    row = {c: 0 for c in feature_cols}
    row["Age"] = age
    row["Gender_Female"] = 1 if gender == "Female" else 0
    row["CGPA"] = cgpa
    row["Sleep_Duration"] = sleep
    row["Study_Hours"] = study
    row["Social_Media_Hours"] = social
    row["Physical_Activity"] = activity
    row["Stress_Level"] = stress
    row["Sleep_Deficit"] = sleep_deficit
    row["Total_Daily_Load"] = total_load
    dept_col = f"Department_{department}"
    if dept_col in row:
        row[dept_col] = 1

    input_df = pd.DataFrame([row])[feature_cols]
    input_scaled = input_df.copy()
    input_scaled[num_cols] = scaler.transform(input_df[num_cols])
    prob = model.predict_proba(input_scaled)[0, 1]

    st.markdown("### Estimated result")
    colL, colR = st.columns([1, 1.4])
    with colL:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
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
            title={"text": "Estimated risk score"},
        ))
        st.plotly_chart(plotly_theme(fig, 300), use_container_width=True)
    with colR:
        if prob < 0.33:
            st.success(
                "Lower estimated score based on these inputs. Healthy sleep and manageable "
                "stress are strongly associated with lower scores in this dataset."
            )
        elif prob < 0.66:
            st.warning(
                "Moderate estimated score. Sleep debt, high daily load, or elevated stress "
                "are nudging this estimate upward."
            )
        else:
            st.error(
                "Higher estimated score based on these inputs. Remember this is a simplified "
                "educational model, not a diagnosis — reaching out to a real support resource "
                "is always the right next step if you're concerned."
            )
        st.caption(
            "This estimate reflects patterns in the demo model's training sample only "
            f"(recall ≈ {metrics['recall']*100:.0f}%, precision ≈ {metrics['precision']*100:.0f}% "
            "on its own held-out test split) — like the original project's model, it favors "
            "catching more at-risk cases over being always precise."
        )

st.markdown("---")
st.caption(
    "Built with Streamlit · Based on the Student Depression & Lifestyle EDA + ML project · "
    "For education only — not a substitute for professional mental health advice."
)
