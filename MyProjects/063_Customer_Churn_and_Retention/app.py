import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="SaaS Customer Churn & Retention Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-title {
        color: #64748b;
        font-size: 13px;
        text-transform: uppercase;
        font-weight: 600;
    }
    .metric-value {
        color: #0f172a;
        font-size: 26px;
        font-weight: 700;
        margin-top: 4px;
    }
    .badge-high {
        background-color: #fee2e2;
        color: #dc2626;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-safe {
        background-color: #dcfce7;
        color: #16a34a;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Multi-Path Artifact Loader
# ---------------------------------------------------------
@st.cache_resource
def load_bundle():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(base_dir, "artifacts", "churn_model_bundle.joblib"),
        os.path.join(base_dir, "churn_model_bundle.joblib"),
        "artifacts/churn_model_bundle.joblib",
        "churn_model_bundle.joblib"
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return joblib.load(p)
    return None

bundle = load_bundle()

if bundle is None:
    st.error("Model artifact not found!")
    st.info("Kripya verify karein ki 'artifacts/churn_model_bundle.joblib' file repo ya folder mein available hai.")
    st.stop()

pipeline = bundle['pipeline']
metrics = bundle.get('metrics', {'roc_auc': 0.88, 'churn_rate': 0.28})

# ---------------------------------------------------------
# Header & Performance KPIs
# ---------------------------------------------------------
st.title("🛡️ B2B SaaS Churn & Revenue Retention Engine")
st.markdown("Proactive early warning system to safeguard recurring revenue (MRR) and detect disengagement signals.")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Model ROC-AUC</div><div class="metric-value">{metrics.get("roc_auc", 0.88)}</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Portfolio Baseline Churn</div><div class="metric-value">{metrics.get("churn_rate", 0.28):.1%}</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card"><div class="metric-title">Lead Time Alert</div><div class="metric-value">30-60 Days</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-card"><div class="metric-title">Retention ROI Target</div><div class="metric-value" style="color: #16a34a;">4.5x</div></div>', unsafe_allow_html=True)

st.markdown("---")

# ---------------------------------------------------------
# Tabs: Single Simulator vs Batch Diagnostics
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["🔍 Account Simulator", "📂 Batch Portfolio Diagnosis"])

with tab1:
    col_input, col_result = st.columns([1, 1.2])

    with col_input:
        st.subheader("Customer Behavioral Attributes")
        tenure = st.slider("Customer Tenure (Months)", min_value=1, max_value=60, value=6)
        contract = st.selectbox("Contract Commitment", ["Month-to-Month", "One-Year", "Two-Year"])
        monthly_charges = st.slider("Monthly Contract Value ($)", min_value=20.0, max_value=250.0, value=85.0, step=5.0)
        support_tickets = st.number_input("Support Tickets (Last 60 Days)", min_value=0, max_value=15, value=3)
        usage_drop_pct = st.slider("Usage Velocity (Last 30 Days Drop %)", min_value=-0.80, max_value=0.80, value=-0.25, step=0.05,
                                  help="Negative value indicates drop in platform activity")
        payment_failures = st.selectbox("Payment / Invoice Failures (Trailing 90D)", [0, 1, 2, 3], index=1)

    input_data = pd.DataFrame([{
        'tenure_months': tenure,
        'contract_type': contract,
        'monthly_charges': monthly_charges,
        'support_tickets': support_tickets,
        'usage_drop_pct': usage_drop_pct,
        'payment_failures': payment_failures
    }])

    prob = pipeline.predict_proba(input_data)[0][1]

    with col_result:
        st.subheader("Diagnostic Risk Verdict")
        
        # Risk gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={'suffix': "%"},
            title={'text': "Predicted Churn Probability"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#dc2626" if prob >= 0.50 else ("#f59e0b" if prob >= 0.30 else "#16a34a")},
                'steps': [
                    {'range': [0, 30], 'color': "#f0fdf4"},
                    {'range': [30, 50], 'color': "#fffbeb"},
                    {'range': [50, 100], 'color': "#fef2f2"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 3},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig.update_layout(height=260, margin=dict(l=15, r=15, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

        annual_risk = monthly_charges * 12
        if prob >= 0.50:
            st.error(f"🚨 **High Churn Risk Detected!** Annual ARR at risk: **${annual_risk:,.2f}**")
            st.markdown("""
            **Recommended Interventions:**
            - Executive sponsor check-in call within 48 hours.
            - Offer 15% discount for switching from Month-to-Month to an Annual plan.
            - Prioritize outstanding support tickets to address friction.
            """)
        elif prob >= 0.30:
            st.warning(f"⚠️ **Moderate Friction Account.** Annual ARR: **${annual_risk:,.2f}**")
            st.markdown("""
            **Recommended Interventions:**
            - Send targeted feature-reengagement sequence.
            - Verify auto-renew billing details to prevent involuntary payment failure.
            """)
        else:
            st.success(f"✅ **Account is Healthy & Expanding.** Annual ARR: **${annual_risk:,.2f}**")
            st.markdown("""
            **Recommended Interventions:**
            - Candidate for tier upgrade or annual expansion plan.
            """)

with tab2:
    st.subheader("Batch Customer Risk Audit")
    uploaded_file = st.file_uploader("Upload CSV file for batch inference", type=["csv"])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        required_cols = ['tenure_months', 'contract_type', 'monthly_charges', 'support_tickets', 'usage_drop_pct', 'payment_failures']
        
        missing = [c for c in required_cols if c not in batch_df.columns]
        if missing:
            st.error(f"Uploaded CSV missing required columns: {missing}")
        else:
            probs = pipeline.predict_proba(batch_df[required_cols])[:, 1]
            batch_df['churn_probability'] = np.round(probs, 3)
            batch_df['risk_tier'] = np.where(probs >= 0.50, 'High', np.where(probs >= 0.30, 'Medium', 'Safe'))
            
            high_risk_count = (batch_df['risk_tier'] == 'High').sum()
            total_at_risk_mrr = batch_df[batch_df['risk_tier'] == 'High']['monthly_charges'].sum()
            
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Audited Accounts", len(batch_df))
            m_col2.metric("Critical High-Risk Accounts", high_risk_count)
            m_col3.metric("At-Risk MRR Exposed", f"${total_at_risk_mrr:,.2f}")
            
            st.dataframe(batch_df[['customer_id', 'monthly_charges', 'contract_type', 'churn_probability', 'risk_tier']] if 'customer_id' in batch_df.columns else batch_df[['monthly_charges', 'contract_type', 'churn_probability', 'risk_tier']], use_container_width=True)
    else:
        st.info("Tip: Aap test karne ke liye 'data/saas_customer_churn.csv' file ko upload karke dekh sakte hain.")