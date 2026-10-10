import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go

st.set_page_config(
    page_title="Retail Demand Forecasting Engine",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
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
</style>
""", unsafe_allow_html=True)

# 1. Model Loading
@st.cache_resource
def load_bundle():
    model_path = "artifacts/store_demand_forecaster.joblib"
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

bundle = load_bundle()

# Sidebar Controls
st.sidebar.image("https://img.icons8.com/color/96/000000/shop.png", width=64)
st.sidebar.title("Demand Parameters")
st.sidebar.caption("Retail Inventory & Replenishment Simulator")

if bundle is None:
    st.error("Model artifact not found! Place `store_demand_forecaster.joblib` inside the `artifacts/` folder.")
    st.stop()

stores = bundle['stores']
items = bundle['items']

selected_store = st.sidebar.selectbox("Select Store ID", stores, index=0)
selected_item = st.sidebar.selectbox("Select Item ID", items, index=0)

forecast_date = st.sidebar.date_input("Forecast Date", pd.to_datetime("2024-01-01"))
promo_active = st.sidebar.selectbox("Promotional Event Active?", [0, 1], format_func=lambda x: "Active Promo (1)" if x == 1 else "No Promo (0)")
unit_price = st.sidebar.slider("Unit Price (\$)", min_value=5.0, max_value=60.0, value=25.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.subheader("Recent Sales Trajectory")
lag_1 = st.sidebar.number_input("Yesterday Sales (Lag-1)", min_value=0, max_value=250, value=45)
lag_7 = st.sidebar.number_input("Last Week Same Day (Lag-7)", min_value=0, max_value=250, value=42)
lag_14 = st.sidebar.number_input("Two Weeks Ago (Lag-14)", min_value=0, max_value=250, value=40)
lag_30 = st.sidebar.number_input("One Month Ago (Lag-30)", min_value=0, max_value=250, value=38)

rolling_mean_7 = st.sidebar.number_input("7-Day Moving Avg", min_value=0.0, max_value=250.0, value=43.5)
rolling_std_7 = st.sidebar.number_input("7-Day Std Dev", min_value=0.0, max_value=50.0, value=4.2)
rolling_mean_30 = st.sidebar.number_input("30-Day Moving Avg", min_value=0.0, max_value=250.0, value=41.0)
rolling_std_30 = st.sidebar.number_input("30-Day Std Dev", min_value=0.0, max_value=50.0, value=5.1)

# Main Dashboard
st.title("📦 Store Item Demand Forecasting Engine")
st.markdown("Automated forward-looking unit demand forecasting powered by `HistGradientBoostingRegressor`.")

# Top Metrics Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-card"><div class="metric-title">Test WAPE</div><div class="metric-value">9.51%</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card"><div class="metric-title">Test MAE</div><div class="metric-value">2.52 units</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card"><div class="metric-title">Naive Baseline WAPE</div><div class="metric-value">19.35%</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-card"><div class="metric-title">Error Reduction</div><div class="metric-value" style="color: #059669;">+50.8%</div></div>', unsafe_allow_html=True)

st.markdown("---")

# Feature Construction
dt = pd.to_datetime(forecast_date)
dayofweek = dt.dayofweek
day = dt.day
month = dt.month
quarter = dt.quarter
dayofyear = dt.dayofyear
is_weekend = int(dayofweek >= 5)

input_dict = {
    'store_id': [selected_store],
    'item_id': [selected_item],
    'price': [unit_price],
    'promo': [promo_active],
    'weekday': [dayofweek],
    'month': [month],
    'day': [day],
    'dayofweek': [dayofweek],
    'is_weekend': [is_weekend],
    'quarter': [quarter],
    'dayofyear': [dayofyear],
    'sin_dayofweek': [np.sin(2 * np.pi * dayofweek / 7)],
    'cos_dayofweek': [np.cos(2 * np.pi * dayofweek / 7)],
    'sin_month': [np.sin(2 * np.pi * month / 12)],
    'cos_month': [np.cos(2 * np.pi * month / 12)],
    'lag_1': [lag_1],
    'lag_7': [lag_7],
    'lag_14': [lag_14],
    'lag_30': [lag_30],
    'rolling_mean_7': [rolling_mean_7],
    'rolling_std_7': [rolling_std_7],
    'rolling_mean_30': [rolling_mean_30],
    'rolling_std_30': [rolling_std_30]
}

input_df = pd.DataFrame(input_dict)
for c in bundle['cat_cols']:
    input_df[c] = input_df[c].astype('category')

# Model Prediction
predicted_sales = bundle['model'].predict(input_df)[0]
predicted_sales = max(0.0, predicted_sales)

# Display Results
res_col1, res_col2 = st.columns([1, 1.2])

with res_col1:
    st.subheader("🎯 Forecast Output")
    st.metric(label=f"Predicted Unit Sales for {forecast_date.strftime('%Y-%m-%d')}", value=f"{predicted_sales:.1f} Units")
    
    # Inventory Replenishment Recommendation
    safety_stock = int(np.ceil(predicted_sales + (1.65 * rolling_std_7)))
    recommended_order = max(0, safety_stock - lag_1)
    
    st.info(f"""
    **Operational Supply Chain Guidelines:**
    - **Recommended Buffer / Safety Stock:** `{safety_stock}` units (95% Service Level)
    - **Suggested Daily Replenishment Order:** `{recommended_order}` units
    """)

with res_col2:
    st.subheader("📈 Demand Momentum Context")
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=['30 Days Ago', '14 Days Ago', '7 Days Ago', 'Yesterday', 'Forecast (Target)'],
        y=[lag_30, lag_14, lag_7, lag_1, predicted_sales],
        mode='lines+markers+text',
        name='Sales Velocity',
        text=[f"{lag_30}", f"{lag_14}", f"{lag_7}", f"{lag_1}", f"{predicted_sales:.1f}"],
        textposition="top center",
        line=dict(color='#059669', width=2.5),
        marker=dict(size=8, color=['#0284c7', '#0284c7', '#0284c7', '#0284c7', '#dc2626'])
    ))
    
    fig.update_layout(
        title="Unit Demand Run-Rate to Forecast",
        xaxis_title="Timeline Step",
        yaxis_title="Unit Sales",
        template="plotly_white",
        height=320,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)