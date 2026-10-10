# 📊 B2B SaaS Customer Churn \& Retention Analytics Engine

An end-to-end Machine Learning intelligence engine built to detect early customer disengagement, quantify Monthly Recurring Revenue (MRR) exposure, and recommend automated retention playbooks for SaaS and subscription businesses.

\---

## 🎯 Executive Summary \& Business Impact

For subscription and recurring-revenue business models, acquiring a new account costs **5x to 7x more** than retaining an existing one. Unaddressed churn directly erodes customer lifetime value (LTV) and compounding recurring revenue.

This solution shifts customer success teams from reactive fire-fighting to proactive retention by:

* **Flagging high-risk accounts 30–60 days in advance** of subscription cancellation or contract lapse.
* **Quantifying financial exposure** through an instant Dollar MRR-at-Risk metric.
* **Prescribing tailored interventions** (e.g., proactive technical check-in, commercial discounting, or executive review) based on account-level root causes.

\---

## 🔍 Predictive Signals \& Data Dictionary

The inference pipeline monitors customer health across account lifecycle, commercial commitment, product engagement, and operational friction indicators:

|Feature Name|Type|Description|Strategic Signal|
|-|-|-|-|
|`customer\\\_id`|String|Unique account identifier|Tracking / audit key|
|`tenure\\\_months`|Integer|Account age in months|Lifecycle stability / onboarding health|
|`contract\\\_type`|Categorical|Month-to-Month, 1-Year, 2-Year|Commercial lock-in \& switching costs|
|`monthly\\\_charges`|Float|Active Monthly Recurring Revenue ($)|Financial prioritization tier|
|`support\\\_tickets`|Integer|Volume of support requests|Operational friction \& user frustration|
|`usage\\\_drop\\\_pct`|Float|30-day activity delta vs trailing mean|Silent disengagement marker|
|`payment\\\_failures`|Integer|Historical billing / card decline events|Involuntary churn risk indicator|

\---

## ⚙️ Machine Learning Pipeline \& Validation Strategy

1. **Preprocessing \& Feature Architecture:**

   * Categorical encodings with target-independent validation pipelines.
   * Robust scaling and imputation layers integrated into serialized pipelines to prevent test leakage.
2. **Modeling Technique:**

   * **Algorithm:** Tree-based ensembles (`HistGradientBoostingClassifier` / `RandomForestClassifier`) tuned for tabular interactions and skewed distributions.
3. **Evaluation Framework (Cost-Sensitive Evaluation):**

   * **Primary Metric:** Precision-Recall AUC (PR-AUC) and Recall at calibrated probability thresholds ($\\tau = 0.40$).
   * **Cost Philosophy:** False Negatives (unidentified churners who leave silently) carry significantly higher operational cost than False Positives (loyal accounts receiving proactive retention attention).

\---

## 🖥️ Interactive Decision Support Dashboard (Streamlit)

The web dashboard is designed for Customer Success leads, Account Executives, and Growth operators:

* **Scenario Simulator:** Real-time probability forecasting on individual accounts using dynamic control sliders.
* **Batch CSV Analysis:** Bulk-upload customer rosters to group accounts into High, Medium, and Low risk bands.
* **Executive KPI Summary:** Aggregated view of total accounts monitored, predicted churn count, and cumulative MRR exposed to churn.

\---

## 🛠️ Tech Stack \& Dependencies

* **Language:** Python 3.10+
* **Data Engineering:** `pandas`, `numpy`
* **Machine Learning \& Serialization:** `scikit-learn`, `joblib`
* **Application \& Visualization:** `streamlit`, `plotly`

\---

## 🚀 Quickstart \& Local Execution

### 1\. Clone the repository \& install dependencies

```bash
git clone https://github.com/<your-username>/saas-churn-retention-engine.git
cd saas-churn-retention-engine
pip install -r requirements.txt
```

### 2\. Generate training data \& train the model

```bash
python data\\\_generator.py
python train\\\_churn\\\_model.py
```

### 3\. Launch the dashboard

```bash
streamlit run app.py
```

\---

## 📁 Repository Structure

```text
saas-churn-retention-engine/
├── data/
│   └── saas\\\_customer\\\_churn.csv       # Synthetic historical cohort dataset
├── artifacts/
│   └── churn\\\_model\\\_bundle.joblib     # Serialized model, encoders, and metrics
├── data\\\_generator.py                 # Reproducible cohort generation script
├── train\\\_churn\\\_model.py              # ML training, cross-validation \\\& packaging
├── app.py                            # Streamlit decision support interface
├── requirements.txt                  # Locked production dependencies
└── README.md                         # Project documentation and specifications
```

