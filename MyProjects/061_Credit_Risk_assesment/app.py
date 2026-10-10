import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="Credit Risk Underwriting Engine",
    page_icon="💳",
    layout="wide"
)

# Load production model bundle
@st.cache_resource
def load_model():
    return joblib.load("credit_risk_pipeline.joblib")

try:
    artifact_bundle = load_model()
    model = artifact_bundle["pipeline"]
    optimal_threshold = artifact_bundle["optimal_threshold"]
except Exception as e:
    st.error(f"Error loading model artifact: {e}")
    st.stop()

# Header banner
st.title("💳 Credit Risk & Loan Default Scoring Engine")
st.markdown(
    "Production-grade decision engine evaluating applicant default risk using **HistGradientBoosting** "
    "and cost-calibrated decision thresholds."
)
st.divider()

# Layout: Two input columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Applicant Profile")
    person_age = st.number_input("Age (Years)", min_value=18, max_value=99, value=30, step=1)
    person_income = st.number_input("Annual Income ($)", min_value=1000, max_value=10000000, value=65000, step=1000)
    person_emp_length = st.number_input("Employment Length (Years)", min_value=0.0, max_value=59.0, value=5.0, step=0.5)
    person_home_ownership = st.selectbox("Home Ownership", options=["RENT", "MORTGAGE", "OWN", "OTHER"])
    cb_person_cred_hist_length = st.number_input("Credit History Length (Years)", min_value=1, max_value=40, value=6, step=1)
    cb_person_default_on_file = st.selectbox("Historical Default on File?", options=["N", "Y"])

with col2:
    st.subheader("📋 Loan Specifications")
    loan_amnt = st.number_input("Requested Loan Amount ($)", min_value=500, max_value=100000, value=10000, step=500)
    loan_intent = st.selectbox("Loan Intent", options=[
        "PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"
    ])
    loan_grade = st.selectbox("Assigned Loan Grade", options=["A", "B", "C", "D", "E", "F", "G"])
    loan_int_rate = st.number_input("Interest Rate (%)", min_value=4.0, max_value=30.0, value=11.5, step=0.1)

    # Derived financial metric
    calculated_percent_income = loan_amnt / person_income if person_income > 0 else 0.0
    st.metric("Loan-to-Income Exposure", f"{calculated_percent_income:.1%}")

st.divider()

# Underwriting decision button
if st.button("🚀 Evaluate Underwriting Decision", use_container_width=True):
    # Construct model input payload
    applicant_payload = pd.DataFrame([{
        "person_age": person_age,
        "person_income": person_income,
        "person_home_ownership": person_home_ownership,
        "person_emp_length": person_emp_length,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": round(calculated_percent_income, 3),
        "cb_person_default_on_file": cb_person_default_on_file,
        "cb_person_cred_hist_length": cb_person_cred_hist_length
    }])

    # Compute default probability
    pred_proba = model.predict_proba(applicant_payload)[0, 1]

    # Render decision cards
    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:
        st.metric("Predicted Default Probability", f"{pred_proba:.2%}")
    with res_col2:
        st.metric("Calibrated Cut-Off Threshold", f"{optimal_threshold:.2%}")
    with res_col3:
        if pred_proba >= optimal_threshold:
            st.error("Decision: REJECT / MANUAL UNDERWRITING")
        else:
            st.success("Decision: APPROVE")

    # Risk summary details
    st.markdown("### Risk Diagnostic Assessment")
    if pred_proba >= optimal_threshold:
        st.warning(
            f"⚠️ **High Default Exposure:** The applicant's default probability ({pred_proba:.2%}) exceeds the "
            f"institutional cut-off threshold ({optimal_threshold:.2%}). Primary risk factors typically include high "
            f"loan-to-income ratio ({calculated_percent_income:.1%}), high-risk loan grade ({loan_grade}), or historical default records."
        )
    else:
        st.info(
            f"✅ **Acceptable Credit Profile:** The applicant's default probability ({pred_proba:.2%}) is safely within the "
            f"acceptable threshold ({optimal_threshold:.2%}). Application meets automatic underwriting criteria."
        )