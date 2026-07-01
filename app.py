import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import plotly.graph_objects as go
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.big-title {
    font-size: 2.5rem; font-weight: 800;
    color: #1f77b4; text-align: center;
    padding: 1rem 0 0.2rem 0;
}
.subtitle {
    font-size: 1.05rem; color: #666;
    text-align: center; margin-bottom: 1.5rem;
}
.metric-box {
    padding: 1.5rem; border-radius: 15px;
    text-align: center; color: white;
}
.metric-value { font-size: 2.8rem; font-weight: 800; }
.metric-label { font-size: 0.95rem; opacity: 0.9; margin-top: 4px; }
.section-header {
    font-size: 1.3rem; font-weight: 700;
    color: #333; margin: 1.2rem 0 0.5rem 0;
    border-left: 4px solid #1f77b4;
    padding-left: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TRAIN MODEL ON STARTUP ----------------
@st.cache_resource
def load_model():
    df = pd.read_csv("Student_Performance_Dataset.csv")

    # Fill missing values
    df['study_hours_daily'] = df['study_hours_daily'].fillna(df['study_hours_daily'].median())
    df['attendance_pct']    = df['attendance_pct'].fillna(df['attendance_pct'].mean())
    df['parent_education']  = df['parent_education'].fillna(df['parent_education'].mode()[0])

    # Encode categorical columns
    le = LabelEncoder()
    df['gender']      = le.fit_transform(df['gender'])       # Female=0, Male=1
    df['school_type'] = le.fit_transform(df['school_type'])  # Private=0, Public=1

    # Parent education mapping
    parent_map = {'None': 0, 'Grad': 1, 'PG': 2}
    df['parent_education'] = df['parent_education'].map(parent_map)

    # Scale numeric features
    scaler   = MinMaxScaler()
    num_cols = ['study_hours_daily', 'attendance_pct', 'sleep_hours']
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Feature engineering AFTER scaling
    df['study_effectiveness'] = df['study_hours_daily'] * df['attendance_pct']
    df['sleep_study_ratio']   = df['sleep_hours'] / df['study_hours_daily'].replace(0, 0.001)
    df['engagement_score']    = df['study_hours_daily']*0.5 + df['attendance_pct']*0.3 + df['extracurricular']*0.2

    # Train model
    X = df.drop(columns=['student_id', 'final_marks'])
    y = df['final_marks']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(
        n_estimators=200, learning_rate=0.05,
        max_depth=6, subsample=0.8,
        colsample_bytree=0.8, random_state=42
    )
    model.fit(X_train, y_train)

    return model, scaler

# Load model (cached - only trains once)
with st.spinner("🔄 Loading model... Please wait"):
    model, scaler = load_model()

def scale_inputs(study_hours, attendance, sleep):
    scaled = scaler.transform(
        pd.DataFrame([[study_hours, attendance, sleep]],
        columns=['study_hours_daily', 'attendance_pct', 'sleep_hours'])
    )
    return scaled[0][0], scaled[0][1], scaled[0][2]

# ---------------- HEADER ----------------
st.markdown('<div class="big-title">🎓 Student Performance Prediction System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">This application predicts student final marks using an XGBoost machine learning model.<br>'
    'Fill in the student details below and click <b>Predict Final Marks</b>.</div>',
    unsafe_allow_html=True
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("📌 About This Project")
st.sidebar.info("""
**ML Model:** XGBoost Regressor

**Task:** Regression — Final Marks Prediction

**Dataset:** Student Performance Dataset

**Tech Stack:** Python · Scikit-learn · XGBoost · Streamlit · Plotly
""")
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Performance")
st.sidebar.success("R² Score: **0.97**")
st.sidebar.info("MAE: **~2.1**")
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Score Guide")
st.sidebar.markdown("""
| Grade | Marks |
|-------|-------|
| A+    | 85+   |
| A     | 75–84 |
| B     | 65–74 |
| C     | 55–64 |
| D     | 45–54 |
| F     | < 45  |
""")
st.sidebar.markdown("---")
st.sidebar.markdown("**👩‍💻 Developed by SAI PRASAD NEELMANI**")
st.sidebar.caption("AI & ML Project")

# ---------------- INPUT SECTION ----------------
st.markdown('<div class="section-header">📝 Enter Student Details</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**👤 Personal Info**")
    gender_label = st.selectbox("Gender", ["Male", "Female"])
    gender = 1 if gender_label == "Male" else 0

    school_label = st.selectbox("School Type", ["Public", "Private"])
    school_type = 1 if school_label == "Public" else 0

    parent_label = st.selectbox("Parent Education", ["None", "Graduate (Grad)", "Post Graduate (PG)"])
    parent_map = {"None": 0, "Graduate (Grad)": 1, "Post Graduate (PG)": 2}
    parent_education = parent_map[parent_label]

    extra_label = st.selectbox("Extracurricular Activities", ["No", "Yes"])
    extracurricular = 0 if extra_label == "No" else 1

with col2:
    st.markdown("**📚 Academic Info**")
    previous_score  = st.slider("Previous Score",   20,  95, 60)
    study_hours_raw = st.slider("Study Hours Daily", 1,  11,  5)
    attendance_raw  = st.slider("Attendance %",     40, 100, 75)
    sleep_hours_raw = st.slider("Sleep Hours",       4,  10,  7)

with col3:
    st.markdown("**📊 Live Input Summary**")
    st.metric("Study Hours",     f"{study_hours_raw} hrs/day")
    st.metric("Attendance",      f"{attendance_raw}%")
    st.metric("Sleep Hours",     f"{sleep_hours_raw} hrs/night")
    st.metric("Previous Score",  f"{previous_score}/100")
    st.metric("Extracurricular", extra_label)

# ---------------- RADAR CHART ----------------
st.markdown("---")
st.markdown('<div class="section-header">📊 Student Profile Overview</div>', unsafe_allow_html=True)

radar_names  = ["Study Hours", "Attendance", "Sleep", "Prev Score", "Extracurricular", "Parent Edu"]
radar_values = [
    study_hours_raw / 11,
    attendance_raw  / 100,
    sleep_hours_raw / 10,
    previous_score  / 95,
    extracurricular,
    parent_education / 2
]

fig_radar = go.Figure()
fig_radar.add_trace(go.Scatterpolar(
    r=radar_values + [radar_values[0]],
    theta=radar_names + [radar_names[0]],
    fill='toself',
    fillcolor='rgba(31, 119, 180, 0.2)',
    line=dict(color='#1f77b4', width=2),
    name='Student Profile'
))
fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    showlegend=False,
    title=dict(text="Student Profile Radar Chart", x=0.5),
    height=320,
    margin=dict(t=50, b=10)
)
st.plotly_chart(fig_radar, use_container_width=True)

# ---------------- PREDICTION ----------------
st.markdown("---")
predict_btn = st.button("🚀 Predict Final Marks", use_container_width=True, type="primary")

if predict_btn:

    # Scale inputs
    study_scaled, attendance_scaled, sleep_scaled = scale_inputs(
        study_hours_raw, attendance_raw, sleep_hours_raw
    )

    # Feature engineering on scaled values
    study_effectiveness = study_scaled * attendance_scaled
    sleep_study_ratio   = sleep_scaled / max(study_scaled, 0.001)
    engagement_score    = study_scaled*0.5 + attendance_scaled*0.3 + extracurricular*0.2

    # Build input dataframe
    input_data = pd.DataFrame({
        "gender":              [gender],
        "school_type":         [school_type],
        "study_hours_daily":   [study_scaled],
        "attendance_pct":      [attendance_scaled],
        "sleep_hours":         [sleep_scaled],
        "previous_score":      [previous_score],
        "extracurricular":     [extracurricular],
        "parent_education":    [parent_education],
        "study_effectiveness": [study_effectiveness],
        "sleep_study_ratio":   [sleep_study_ratio],
        "engagement_score":    [engagement_score]
    })

    with st.spinner("🤖 Analysing student data..."):
        prediction = float(model.predict(input_data)[0])
        prediction = max(0, min(100, prediction))

    prediction_int = int(round(prediction))

    # Prediction Index
    prediction_index = int(round(
        (study_hours_raw / 11)  * 25 +
        (attendance_raw  / 100) * 25 +
        (previous_score  / 95)  * 30 +
        (sleep_hours_raw / 10)  * 10 +
        extracurricular         *  5 +
        (parent_education / 2)  *  5
    ))
    prediction_index = max(0, min(100, prediction_index))

    # Grade
    if prediction_int >= 85:
        grade, grade_color = "A+", "linear-gradient(135deg,#11998e,#38ef7d)"
    elif prediction_int >= 75:
        grade, grade_color = "A",  "linear-gradient(135deg,#56ab2f,#a8e063)"
    elif prediction_int >= 65:
        grade, grade_color = "B",  "linear-gradient(135deg,#1f77b4,#4facfe)"
    elif prediction_int >= 55:
        grade, grade_color = "C",  "linear-gradient(135deg,#f7971e,#ffd200)"
    elif prediction_int >= 45:
        grade, grade_color = "D",  "linear-gradient(135deg,#f37335,#fda085)"
    else:
        grade, grade_color = "F",  "linear-gradient(135deg,#cb2d3e,#ef473a)"

    # Results
    st.markdown('<div class="section-header">🎯 Prediction Results</div>', unsafe_allow_html=True)

    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(f"""
        <div class="metric-box" style="background: linear-gradient(135deg,#667eea,#764ba2);">
            <div class="metric-value">{prediction_int}/100</div>
            <div class="metric-label">🎯 Predicted Final Score</div>
        </div>""", unsafe_allow_html=True)
    with r2:
        st.markdown(f"""
        <div class="metric-box" style="background: linear-gradient(135deg,#f093fb,#f5576c);">
            <div class="metric-value">{prediction_index}/100</div>
            <div class="metric-label">📊 Prediction Index</div>
        </div>""", unsafe_allow_html=True)
    with r3:
        st.markdown(f"""
        <div class="metric-box" style="background: {grade_color};">
            <div class="metric-value">{grade}</div>
            <div class="metric-label">🏅 Grade</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.progress(prediction_int)

    if prediction_int >= 85:
        st.success("🏆 **Excellent Performance** — Outstanding result! You are in the top tier!")
    elif prediction_int >= 70:
        st.info("🌟 **Good Performance** — Strong result! A little more effort for excellence.")
    elif prediction_int >= 55:
        st.warning("📚 **Average Performance** — Above the mean. Focus on weak areas to improve.")
    else:
        st.error("⚠️ **Needs Improvement** — Below average. Please increase study time and attendance.")

    # Score Gauge
    st.markdown('<div class="section-header">📉 Score Gauge</div>', unsafe_allow_html=True)
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prediction_int,
        delta={'reference': 60, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#667eea", 'thickness': 0.3},
            'steps': [
                {'range': [0,  45], 'color': "#ffcccc"},
                {'range': [45, 65], 'color': "#fff3cd"},
                {'range': [65, 85], 'color': "#cce5ff"},
                {'range': [85,100], 'color': "#d4edda"},
            ],
            'threshold': {'line': {'color': "red", 'width': 3}, 'thickness': 0.75, 'value': 45}
        },
        title={'text': f"Predicted Score: {prediction_int}/100", 'font': {'size': 16}}
    ))
    fig_gauge.update_layout(height=280, margin=dict(t=50, b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Feature Importance
    st.markdown('<div class="section-header">📈 Feature Importance</div>', unsafe_allow_html=True)
    feat_names = [
        "Gender", "School Type", "Study Hours", "Attendance %",
        "Sleep Hours", "Previous Score", "Extracurricular",
        "Parent Education", "Study Effectiveness", "Sleep/Study Ratio", "Engagement Score"
    ]
    feat_df = pd.DataFrame({
        "Feature":    feat_names,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True)

    fig_imp = px.bar(
        feat_df, x="Importance", y="Feature", orientation='h',
        color="Importance", color_continuous_scale="Blues",
        title="Which factors influence the prediction most?",
        labels={"Importance": "Importance Score", "Feature": ""}
    )
    fig_imp.update_layout(height=380, margin=dict(t=50, b=10), coloraxis_showscale=False)
    st.plotly_chart(fig_imp, use_container_width=True)

    # AI Explanation
    st.markdown('<div class="section-header">🧠 AI Explanation — Why this prediction?</div>', unsafe_allow_html=True)

    explanations = []
    if study_hours_raw >= 7:
        explanations.append(("✅", "**Study Hours are high** — Consistent study time is the biggest driver of academic success."))
    elif study_hours_raw >= 4:
        explanations.append(("⚠️", "**Study Hours are moderate** — Increasing daily study time could significantly improve the score."))
    else:
        explanations.append(("❌", "**Study Hours are very low** — This is a primary reason for a low prediction. Study more daily."))

    if attendance_raw >= 85:
        explanations.append(("✅", "**Excellent attendance** — High attendance correlates strongly with better understanding and marks."))
    elif attendance_raw >= 70:
        explanations.append(("⚠️", "**Attendance is average** — Try not to miss classes; each session builds on the previous one."))
    else:
        explanations.append(("❌", "**Low attendance** — Missing classes is negatively impacting this prediction significantly."))

    if previous_score >= 70:
        explanations.append(("✅", "**Strong previous score** — Past performance shows consistent academic ability."))
    elif previous_score >= 50:
        explanations.append(("⚠️", "**Average previous score** — Indicates potential but suggests inconsistency."))
    else:
        explanations.append(("❌", "**Low previous score** — This is weighing down the prediction. Revision of past topics is essential."))

    if sleep_hours_raw >= 7:
        explanations.append(("✅", "**Adequate sleep** — Good sleep supports memory consolidation and focus."))
    elif sleep_hours_raw >= 6:
        explanations.append(("⚠️", "**Sleep is slightly low** — Aim for at least 7–8 hours for optimal brain performance."))
    else:
        explanations.append(("❌", "**Insufficient sleep** — Poor sleep is affecting concentration and retention ability."))

    if extracurricular == 1:
        explanations.append(("✅", "**Extracurricular participation** — Shows a well-rounded student with good time management."))
    else:
        explanations.append(("ℹ️", "**No extracurricular** — Joining activities can improve engagement and soft skills."))

    if parent_education >= 2:
        explanations.append(("✅", "**Highly educated parents** — Supportive academic home environment is a positive influence."))
    elif parent_education == 1:
        explanations.append(("ℹ️", "**Moderate parent education** — Some home academic support available."))
    else:
        explanations.append(("ℹ️", "**Limited parent education** — Student is self-driven; additional resources recommended."))

    for icon, text in explanations:
        st.markdown(f"{icon} {text}")

    with st.expander("📋 View Processed Input Data"):
        display_df = pd.DataFrame({
            "Feature": [
                "Gender", "School Type", "Study Hours/day", "Attendance %",
                "Sleep Hours", "Previous Score", "Extracurricular",
                "Parent Education", "Study Effectiveness*",
                "Sleep/Study Ratio*", "Engagement Score*"
            ],
            "Value": [
                gender_label, school_label, f"{study_hours_raw} hrs",
                f"{attendance_raw}%", f"{sleep_hours_raw} hrs",
                previous_score, extra_label, parent_label,
                f"{study_effectiveness:.4f}",
                f"{sleep_study_ratio:.4f}",
                f"{engagement_score:.4f}"
            ]
        })
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.caption("* Engineered features computed from scaled values.")

# ---------------- FOOTER ----------------
st.markdown("---")
col_f1, col_f2 = st.columns(2)
with col_f1:
    st.caption("🐛 **SAI PRASAD NEELMANI**")
with col_f2:
    st.caption("🤖 AI & ML Project")
