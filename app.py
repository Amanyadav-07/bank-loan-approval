import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Loan Approval Intelligence System",
    page_icon="🏛️",
    layout="wide"
)

# ── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
.stApp {
    background: linear-gradient(160deg, #0f1923 0%, #131f2e 40%, #0f1923 100%);
    color: #e8eaf6;
}
[data-testid="stSidebar"] {
    background: #0d1520 !important;
    border-right: 1px solid rgba(79, 195, 247, 0.15);
    padding-top: 20px;
}
[data-testid="stSidebar"] > div {
    padding: 20px 16px;
}
[data-testid="stSidebar"] h1 {
    color: #ffffff !important;
    font-size: 20px !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(79, 195, 247, 0.2);
    margin-bottom: 20px !important;
}
[data-testid="stSidebar"] .stRadio > div {
    gap: 6px !important;
}
[data-testid="stSidebar"] .stRadio label {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(79, 195, 247, 0.08) !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    color: #a8b8d0 !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    cursor: pointer;
    width: 100%;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(79, 195, 247, 0.08) !important;
    border-color: rgba(79, 195, 247, 0.3) !important;
    color: #4fc3f7 !important;
}
h1 {
    background: linear-gradient(90deg, #ffffff 0%, #81d4fa 60%, #4fc3f7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
}
h2, h3 {
    color: #cce7f5 !important;
    font-weight: 600 !important;
}
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
    background: rgba(255,255,255,0.02);
    border-radius: 14px;
}
[data-testid="stSlider"] {
    background: rgba(13, 33, 55, 0.6) !important;
    border: 1px solid rgba(79, 195, 247, 0.1) !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
}
[data-testid="stSlider"] label {
    color: #7b9db5 !important;
    font-size: 12px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    font-weight: 600 !important;
}
[data-testid="stNumberInput"] {
    background: rgba(13, 33, 55, 0.6) !important;
    border: 1px solid rgba(79, 195, 247, 0.1) !important;
    border-radius: 10px !important;
    padding: 10px 16px !important;
}
[data-testid="stNumberInput"] label {
    color: #7b9db5 !important;
    font-size: 12px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    font-weight: 600 !important;
}
[data-testid="stNumberInput"] input {
    background: #0d2137 !important;
    color: #e8eaf6 !important;
    border: 1px solid rgba(79, 195, 247, 0.2) !important;
    border-radius: 8px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
}
[data-testid="stNumberInput"] > div {
    background: #0d2137 !important;
    border: 1px solid rgba(79, 195, 247, 0.2) !important;
    border-radius: 8px !important;
}
[data-testid="stNumberInput"] button {
    background: #1e3a5f !important;
    color: #4fc3f7 !important;
    border: none !important;
}
[data-testid="stSelectbox"] {
    background: rgba(13, 33, 55, 0.6) !important;
    border: 1px solid rgba(79, 195, 247, 0.1) !important;
    border-radius: 10px !important;
    padding: 10px 16px !important;
}
[data-testid="stSelectbox"] label {
    color: #7b9db5 !important;
    font-size: 12px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    font-weight: 600 !important;
}
[data-testid="stSelectbox"] > div > div {
    background: #0d2137 !important;
    border: 1px solid rgba(79, 195, 247, 0.2) !important;
    border-radius: 8px !important;
    color: #e8eaf6 !important;
}
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0d2137 0%, #132840 100%);
    border: 1px solid rgba(79, 195, 247, 0.15);
    border-radius: 14px;
    padding: 20px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: transform 0.2s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
}
[data-testid="stMetricLabel"] {
    color: #7b9db5 !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 1.2px !important;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
    color: #4fc3f7 !important;
    font-size: 26px !important;
    font-weight: 700 !important;
}
.stButton > button {
    background: linear-gradient(90deg, #1565c0 0%, #1976d2 50%, #1e88e5 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 16px 28px !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    letter-spacing: 0.8px !important;
    box-shadow: 0 4px 20px rgba(21, 101, 192, 0.35) !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 28px rgba(21, 101, 192, 0.55) !important;
    transform: translateY(-2px) !important;
}
hr {
    border: none !important;
    border-top: 1px solid rgba(79, 195, 247, 0.1) !important;
    margin: 28px 0 !important;
}
[data-testid="stInfo"] {
    background: rgba(13, 33, 55, 0.8) !important;
    border-left: 3px solid #4fc3f7 !important;
    border-radius: 0 10px 10px 0 !important;
    color: #a8b8d0 !important;
    font-size: 13px !important;
}
[data-testid="stCaptionContainer"] {
    color: #556b7d !important;
    font-size: 13px !important;
}
.stCaption, footer {
    color: #3a5068 !important;
}
.js-plotly-plot {
    border-radius: 12px !important;
    overflow: hidden !important;
}
</style>
""", unsafe_allow_html=True)

# ── LOAD ARTIFACTS ────────────────────────────────────────────
@st.cache_resource
def load_model_and_artifacts():
    model = joblib.load('loan_approval_model.pkl')
    explainer = joblib.load('shap_explainer.pkl')
    feature_names = joblib.load('feature_names.pkl')
    return model, explainer, feature_names

model, explainer, feature_names = load_model_and_artifacts()

# ── LOAD SQL DATA ─────────────────────────────────────────────
@st.cache_data
def load_sql_data():
    q1 = pd.read_csv("Data/sql_q1_education.csv")
    q2 = pd.read_csv("Data/sql_q2_loan_purpose.csv")
    q3 = pd.read_csv("Data/sql_q3_home_ownership.csv")
    q4 = pd.read_csv("Data/sql_q4_income_band.csv")
    q5 = pd.read_csv("Data/sql_q5_defaults.csv")
    return q1, q2, q3, q4, q5

q1, q2, q3, q4, q5 = load_sql_data()

# ── LOAD CLEANED DATA ─────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("Data/loan_data_cleaned.csv")

df = load_data()

# ── NAVIGATION ────────────────────────────────────────────────
st.sidebar.title("🏛️ Loan Intelligence")
page = st.sidebar.radio("Navigate", ["🔮 Predict Approval", "📊 EDA Insights", "🗄️ SQL Analysis"])

# ── SHARED CHART LAYOUT ───────────────────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(13,27,42,0.8)',
    font=dict(color='#a8b8d0'),
    title_font=dict(color='#81d4fa', size=15),
    xaxis=dict(gridcolor='#1e3a5f'),
    yaxis=dict(gridcolor='#1e3a5f'),
)

# ══════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT
# ══════════════════════════════════════════════════════════════
if page == "🔮 Predict Approval":

    st.title("🏛️ Bank Loan Approval Intelligence System")
    st.markdown("#### Real-Time Loan Decision Engine with SHAP Explainability")
    st.divider()

    st.subheader("📋 Borrower Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        person_age = st.slider("Age", 18, 80, 30)
        person_income = st.number_input("Annual Income ($)", 8000, 500000, 60000, step=1000)
        person_emp_exp = st.slider("Employment Experience (Years)", 0, 50, 5)
        credit_score = st.slider("Credit Score", 390, 850, 650)

    with col2:
        loan_amnt = st.number_input("Loan Amount ($)", 500, 35000, 10000, step=500)
        loan_int_rate = st.slider("Interest Rate (%)", 5.0, 20.0, 11.0, step=0.1)
        loan_percent_income = round(loan_amnt / person_income, 2) if person_income > 0 else 0.0
        st.metric("Loan % of Income (auto)", f"{loan_percent_income:.2%}")
        cb_person_cred_hist_length = st.slider("Credit History Length (Years)", 2, 30, 5)

    with col3:
        person_education = st.selectbox("Education", ["High School", "Associate", "Bachelor", "Master", "Doctorate"])
        person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
        loan_intent = st.selectbox("Loan Purpose", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
        previous_loan_defaults_on_file = st.selectbox("Previous Loan Default?", ["No", "Yes"])

    # Engineered features
    loan_to_income = loan_amnt / person_income if person_income > 0 else 0
    income_per_exp_year = person_income / (person_emp_exp + 1)
    credit_to_loan = credit_score / loan_amnt if loan_amnt > 0 else 0
    previous_default_encoded = 1 if previous_loan_defaults_on_file == "Yes" else 0

    input_df = pd.DataFrame([{
        'person_age': person_age,
        'person_education': person_education,
        'person_income': person_income,
        'person_emp_exp': person_emp_exp,
        'person_home_ownership': person_home_ownership,
        'loan_amnt': loan_amnt,
        'loan_intent': loan_intent,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'cb_person_cred_hist_length': cb_person_cred_hist_length,
        'credit_score': credit_score,
        'previous_loan_defaults_on_file': previous_default_encoded,
        'loan_to_income': loan_to_income,
        'income_per_exp_year': income_per_exp_year,
        'credit_score_to_loan_ratio': credit_to_loan,
    }])

    if st.button("🔍 Assess Loan Application", use_container_width=True):

        prob = model.predict_proba(input_df)[0][1]
        decision = "✅ APPROVED" if prob >= 0.5 else "❌ REJECTED"
        risk = "🟢 Low Risk" if prob >= 0.7 else "🟡 Medium Risk" if prob >= 0.4 else "🔴 High Risk"
        card_color = "#0d3b1e" if prob >= 0.5 else "#3b0d0d"
        border_color = "#2ecc71" if prob >= 0.5 else "#e74c3c"

        st.divider()

        # Result card
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {card_color}, #0d1b2a);
            border: 2px solid {border_color};
            border-radius: 16px;
            padding: 28px 32px;
            margin: 16px 0;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
                <div>
                    <p style="color: #7b9db5; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; margin: 0;">Decision</p>
                    <p style="color: {border_color}; font-size: 36px; font-weight: 800; margin: 4px 0;">{decision}</p>
                </div>
                <div>
                    <p style="color: #7b9db5; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; margin: 0;">Approval Probability</p>
                    <p style="color: #4fc3f7; font-size: 48px; font-weight: 800; margin: 4px 0;">{prob:.1%}</p>
                </div>
                <div>
                    <p style="color: #7b9db5; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; margin: 0;">Risk Level</p>
                    <p style="color: #e8eaf6; font-size: 28px; font-weight: 700; margin: 4px 0;">{risk}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Progress bar
        st.markdown("**Approval Probability Meter**")
        bar_color = "#2ecc71" if prob >= 0.7 else "#f39c12" if prob >= 0.4 else "#e74c3c"
        st.markdown(f"""
        <div style="background: #1e3a5f; border-radius: 50px; height: 22px; margin: 8px 0 20px 0; overflow: hidden;">
            <div style="
                width: {prob*100:.1f}%;
                height: 100%;
                background: linear-gradient(90deg, {bar_color}88, {bar_color});
                border-radius: 50px;
                display: flex;
                align-items: center;
                justify-content: flex-end;
                padding-right: 10px;
            ">
                <span style="color: white; font-size: 12px; font-weight: 700;">{prob*100:.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.info("📌 **About this model:** Previous loan default history is the strongest rejection signal (0% approval for prior defaulters). Higher interest rates paradoxically indicate approval — banks price in risk rather than reject outright.")

        # SHAP chart
        st.divider()
        st.subheader("🧠 Why this decision? (SHAP Explanation)")
        st.caption("Each bar shows how much a feature pushed the prediction toward Approved (+) or Rejected (−)")

        input_transformed = model.named_steps['preprocessor'].transform(input_df)
        shap_vals = explainer.shap_values(input_transformed)[0]

        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP Value': shap_vals
        }).sort_values('SHAP Value', key=abs, ascending=False).head(10)

        shap_df['Direction'] = shap_df['SHAP Value'].apply(
            lambda x: 'Pushes toward Approved' if x > 0 else 'Pushes toward Rejected'
        )

        fig = px.bar(
            shap_df,
            x='SHAP Value',
            y='Feature',
            color='Direction',
            orientation='h',
            color_discrete_map={
                'Pushes toward Approved': '#2ecc71',
                'Pushes toward Rejected': '#e74c3c'
            },
            title="Top 10 Factors Influencing This Decision"
        )
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=450,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(13,27,42,0.8)',
            font=dict(color='#a8b8d0'),
            title_font=dict(color='#81d4fa', size=16),
            xaxis=dict(gridcolor='#1e3a5f'),
            yaxis_title=None,
            legend=dict(
                bgcolor='rgba(13,27,42,0.8)',
                bordercolor='#1e3a5f',
                borderwidth=1
            )
        )
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ══════════════════════════════════════════════════════════════
elif page == "📊 EDA Insights":

    st.title("📊 Exploratory Data Analysis")
    st.markdown("Visual insights from 44,500+ loan applications")
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Applications", f"{len(df):,}")
    col2.metric("Approval Rate", f"{df['loan_status'].mean():.2%}")
    col3.metric("Avg Credit Score", f"{df['credit_score'].mean():.0f}")
    col4.metric("Avg Loan Amount", f"${df['loan_amnt'].mean():,.0f}")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Approval Rate by Home Ownership")
        home_approval = df.groupby('person_home_ownership')['loan_status'].mean().reset_index()
        home_approval.columns = ['Home Ownership', 'Approval Rate']
        home_approval['Approval Rate'] = (home_approval['Approval Rate'] * 100).round(2)
        fig1 = px.bar(home_approval, x='Home Ownership', y='Approval Rate',
                      color='Approval Rate', color_continuous_scale='blues',
                      title="Approval Rate % by Home Ownership")
        fig1.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Loan Purpose Distribution")
        intent_counts = df['loan_intent'].value_counts().reset_index()
        intent_counts.columns = ['Loan Purpose', 'Count']
        fig2 = px.pie(intent_counts, names='Loan Purpose', values='Count',
                      title="Applications by Loan Purpose")
        fig2.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Credit Score Distribution by Outcome")
        fig3 = px.histogram(df, x='credit_score', color='loan_status',
                            barmode='overlay', nbins=50,
                            color_discrete_map={0: '#e74c3c', 1: '#2ecc71'},
                            labels={'loan_status': 'Status', 'credit_score': 'Credit Score'},
                            title="Credit Score: Approved vs Rejected")
        fig3.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        st.subheader("Income vs Loan Amount")
        sample = df.sample(n=2000, random_state=42)
        sample['loan_status_label'] = sample['loan_status'].map({0: 'Rejected', 1: 'Approved'})
        fig4 = px.scatter(sample, x='person_income', y='loan_amnt',
                          color='loan_status_label',
                          color_discrete_map={'Rejected': '#e74c3c', 'Approved': '#2ecc71'},
                          labels={'loan_status_label': 'Status', 'person_income': 'Income', 'loan_amnt': 'Loan Amount'},
                          title="Income vs Loan Amount (sample 2000)",
                          opacity=0.6)
        fig4.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 3 — SQL
# ══════════════════════════════════════════════════════════════
elif page == "🗄️ SQL Analysis":

    st.title("🗄️ SQL Business Insights")
    st.markdown("5 business questions answered using SQLite queries on 44,500+ records")
    st.divider()

    st.subheader("Which loan purpose gets rejected most?")
    fig_q2 = px.bar(q2.sort_values('rejection_rate_pct', ascending=True),
                    x='rejection_rate_pct', y='loan_intent', orientation='h',
                    color='rejection_rate_pct', color_continuous_scale='reds',
                    title="Rejection Rate % by Loan Purpose",
                    labels={'rejection_rate_pct': 'Rejection Rate %', 'loan_intent': 'Loan Purpose'})
    fig_q2.update_layout(**CHART_LAYOUT)
    st.plotly_chart(fig_q2, use_container_width=True)

    st.subheader("Does higher income mean higher approval?")
    st.caption("Counterintuitive finding: lower income borrowers get approved more — they take smaller loans.")
    fig_q4 = px.bar(q4, x='income_band', y='approval_rate_pct',
                    color='approval_rate_pct', color_continuous_scale='greens',
                    title="Approval Rate % by Income Band",
                    labels={'approval_rate_pct': 'Approval Rate %', 'income_band': 'Income Band'})
    fig_q4.update_layout(**CHART_LAYOUT)
    st.plotly_chart(fig_q4, use_container_width=True)

    st.subheader("💣 The most powerful finding: Previous Defaults")
    col1, col2 = st.columns(2)
    with col1:
        fig_q5 = px.bar(q5, x='previous_loan_defaults_on_file', y='approval_rate_pct',
                        color='approval_rate_pct', color_continuous_scale='RdYlGn',
                        title="Approval Rate: Previous Default vs No Default",
                        labels={'approval_rate_pct': 'Approval Rate %',
                                'previous_loan_defaults_on_file': 'Previous Default (1=Yes, 0=No)'})
        fig_q5.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig_q5, use_container_width=True)
    with col2:
        st.metric("Approval rate — No previous default", f"{q5[q5['previous_loan_defaults_on_file']==0]['approval_rate_pct'].values[0]:.1f}%")
        st.metric("Approval rate — Previous defaulter", f"{q5[q5['previous_loan_defaults_on_file']==1]['approval_rate_pct'].values[0]:.1f}%")
        st.info("A single previous default results in **0% approval rate** across all 22,593 applicants with a default history. This is the strongest single predictor in the entire model.")

    st.subheader("Credit Score by Home Ownership & Approval")
    fig_q3 = px.bar(q3, x='person_home_ownership', y='avg_credit_score',
                    color='loan_status', barmode='group',
                    color_discrete_map={0: '#e74c3c', 1: '#2ecc71'},
                    title="Avg Credit Score by Home Ownership and Approval Status",
                    labels={'avg_credit_score': 'Avg Credit Score',
                            'person_home_ownership': 'Home Ownership',
                            'loan_status': 'Approved'})
    fig_q3.update_layout(**CHART_LAYOUT)
    st.plotly_chart(fig_q3, use_container_width=True)

    st.subheader("Does education level matter?")
    st.caption("Finding: Education has almost no impact on approval rate — all levels cluster around 22%.")
    st.dataframe(q1, use_container_width=True)

# ── FOOTER ────────────────────────────────────────────────────
st.divider()
st.caption("© 2026 Bank Loan Approval Intelligence System | Built by Aman Yadav | Python · XGBoost · SHAP · Streamlit")