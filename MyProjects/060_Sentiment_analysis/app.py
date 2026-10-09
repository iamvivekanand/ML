import streamlit as st
import joblib
import re

# 1. Page Configuration
st.set_page_config(
    page_title="Product Review & Feedback Analyzer",
    layout="centered"
)

# 2. Custom Styling (Background Color, Border & Clean Card Layout)
st.markdown("""
    <style>
    /* Full screen viewport background */
    [data-testid="stAppViewContainer"] {
        background-color: #f1f5f9 !important;
    }

    /* Main content card container with prominent border */
    .block-container {
        max-width: 740px !important;
        padding-top: 2.5rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
        background-color: #ffffff !important;
        border-radius: 14px !important;
        border: 2px solid #cbd5e1 !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.07) !important;
        margin-top: 2.5rem !important;
        margin-bottom: 2.5rem !important;
    }

    /* Clean header styling */
    h1 {
        font-size: 1.85rem !important;
        font-weight: 700 !important;
        color: #0f172a !important;
        margin-bottom: 0.2rem !important;
    }

    /* Subtitle styling */
    .sub-text {
        color: #64748b;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Load Trained Model and Vectorizer
@st.cache_resource
def load_assets():
    model = joblib.load('best_sentiment_model.pkl')
    vectorizer = joblib.load('best_tfidf_vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_assets()

# 4. Text Preprocessing Function
def clean_text(text):
    text = str(text).lower()
    text = text.replace('_', ' ')
    text = re.sub(r'[^a-z\s]', '', text)
    return ' '.join(text.split())

# 5. Issue Categorization Logic for Negative Feedback
def tag_issue(text):
    text = text.lower()
    if any(k in text for k in ['battery', 'charge', 'power', 'cable', 'screen', 'sound', 'button']):
        return 'Hardware & Battery'
    elif any(k in text for k in ['deliver', 'delivery', 'late', 'ship', 'shipping', 'delay', 'arrive']):
        return 'Shipping & Logistics'
    elif any(k in text for k in ['cheap', 'break', 'broken', 'plastic', 'poor', 'crack', 'damage', 'scratch']):
        return 'Build Quality & Packaging'
    elif any(k in text for k in ['price', 'expensive', 'cost', 'waste', 'money', 'worth']):
        return 'Pricing & Value'
    elif any(k in text for k in ['service', 'support', 'return', 'refund', 'warranty', 'help']):
        return 'Customer Support & Warranty'
    else:
        return 'General Product Defect'

# 6. UI Header without Emojis
st.title("Product Review & Feedback Analyzer")
st.markdown('<p class="sub-text">Live sentiment classification and root-cause issue detection pipeline.</p>', unsafe_allow_html=True)

# User input text area
user_review = st.text_area(
    "Customer Review Text:",
    placeholder="Type or paste a product review here... (e.g. 'The battery stopped working after 3 days')",
    height=130
)

# Analyze action button
if st.button("Analyze Review", type="primary"):
    if user_review.strip():
        # Preprocess and score sentiment
        cleaned = clean_text(user_review)
        vec_input = vectorizer.transform([cleaned])
        prediction = model.predict(vec_input)[0]

        st.divider()

        # Display result card
        if prediction == 'Positive':
            st.success("**Sentiment Detected:** Positive")
            st.info("Customer feedback indicates a satisfactory product experience.")
        else:
            issue = tag_issue(cleaned)
            st.error("**Sentiment Detected:** Negative")
            st.warning(f"**Root Cause Category:** {issue}")
            st.caption("Action: Flagged for operations and quality-audit triage.")
    else:
        st.warning("Please enter a review first.")