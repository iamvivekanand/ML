\# Credit Risk \& Loan Default Scoring Engine



\[!\[Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

\[!\[Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)

\[!\[License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



An institutional-grade credit underwriting and default risk classification pipeline. This engine combines leak-proof preprocessing, non-linear gradient-boosted ensembles, and cost-sensitive threshold optimization to minimize bad-debt write-offs in consumer credit portfolios.



\---



\## 1. Executive Summary \& Business Impact



In retail lending, default classification incurs asymmetric operational costs: approving a borrower who subsequently defaults (\*\*False Negative\*\*) triggers direct principal loss, whereas declining a creditworthy borrower (\*\*False Positive\*\*) causes nominal opportunity loss.



\- \*\*Primary Objective:\*\* Score applicant probability of default ($P$) and optimize capital allocation decisions.

\- \*\*Discriminative Performance:\*\* Achieved an out-of-sample \*\*0.9426 ROC-AUC\*\* and \*\*0.8955 PR-AUC\*\* via `HistGradientBoosting`.

\- \*\*Cost Optimization:\*\* Lowering the operational decision threshold from the standard \*\*0.50\*\* to \*\*0.16\*\* yielded:

&#x20; - \*\*82.90% Default Recall\*\* (up from 72.13%).

&#x20; - \*\*15.17% Reduction in Total Portfolio Misclassification Cost\*\* under an asymmetric $5:1$ loss penalty structure.



\---



\## 2. End-to-End Architecture

\[ Raw Applicant Payload / CSV ]

&#x20;                │

&#x20;                ▼

&#x20; ┌────────────────────────────────────────────────────────┐

&#x20; │  Stage 1: Domain Sanitization \& Integrity Checks       │

&#x20; │  • Filter person\_age < 100                             │

&#x20; │  • Filter person\_emp\_length < 60                       │

&#x20; └──────────────────────────────┬─────────────────────────┘

&#x20;                                │

&#x20;                                ▼

&#x20; ┌────────────────────────────────────────────────────────┐

&#x20; │  Stage 2: Modular Preprocessing (ColumnTransformer)     │

&#x20; │  • Numeric Branch    : SimpleImputer(median)           │

&#x20; │                        + StandardScaler()              │

&#x20; │  • Categoric Branch  : SimpleImputer(most\_frequent)    │

&#x20; │                        + OneHotEncoder(sparse=False)   │

&#x20; └──────────────────────────────┬─────────────────────────┘

&#x20;                                │

&#x20;                                ▼

&#x20; ┌────────────────────────────────────────────────────────┐

&#x20; │  Stage 3: Gradient Boosted Predictive Engine           │

&#x20; │  • Algorithm: HistGradientBoostingClassifier           │

&#x20; │  • Computes: P(Default = 1 | X)                        │

&#x20; │  • Metric Baseline: 0.9426 ROC-AUC / 0.8955 PR-AUC     │

&#x20; └──────────────────────────────┬─────────────────────────┘

&#x20;                                │

&#x20;                                ▼

&#x20; ┌────────────────────────────────────────────────────────┐

&#x20; │  Stage 4: Cost-Sensitive Decision Calibration          │

&#x20; │  • Asymmetric Loss Optimization (5:1 Penalty Ratio)    │

&#x20; │  • Threshold Cut-off: 0.1600 (Replaces default 0.50)   │

&#x20; │  • Metric Result: 82.90% Default Recall                │

&#x20; └──────────────────────────────┬─────────────────────────┘

&#x20;                                │

&#x20;                                ▼

&#x20; ┌────────────────────────────────────────────────────────┐

&#x20; │  Stage 5: Production Scoring \& Decision Engine         │

&#x20; │  • Probability >= 0.1600 ──> REJECT / MANUAL REVIEW    │

&#x20; │  • Probability <  0.1600 ──> APPROVE                   │

&#x20; │  • Explainability: Permutation Importance Drivers      │

&#x20; └────────────────────────────────────────────────────────┘



\## 3. Benchmark Model Comparison



All models were evaluated on an out-of-sample stratified test partition ($N = 6,515$ records, 20% holdout):



| Algorithm Family | Test ROC-AUC | Test PR-AUC | Default Recall (@ 0.50) | Execution Latency |

| :--- | :---: | :---: | :---: | :---: |

| \*\*HistGradientBoosting (Selected)\*\* | \*\*0.9426\*\* | \*\*0.8955\*\* | \*\*72.13%\*\* | \*\*\~0.85s\*\* |

| Random Forest (Balanced) | 0.9250 | 0.8705 | 68.42% | \~3.40s |

| Logistic Regression (Balanced Baseline) | 0.8636 | 0.7081 | 79.10% | \~0.35s |



\---



\## 4. Key Financial Risk Drivers



Permutation importance analysis on unseen holdout records isolated the following primary determinants of credit failure:



1\. \*\*`person\_income`\*\* (+0.0893 Delta AUC): Baseline repayment capacity.

2\. \*\*`loan\_percent\_income`\*\* (+0.0885 Delta AUC): Debt-to-income leverage ratio.

3\. \*\*`person\_home\_ownership`\*\* (+0.0838 Delta AUC): Collateral stability and asset foundation.

4\. \*\*`loan\_intent`\*\* (+0.0491 Delta AUC): Specific debt purpose risk (e.g., debt consolidation vs. education).

5\. \*\*`loan\_grade`\*\* (+0.0445 Delta AUC): External credit risk band.



\---



\## 5. Repository Structure



├── app.py                      # Interactive Streamlit underwriting interface

├── credit\_risk\_pipeline.joblib # Serialized model bundle \& calibrated threshold

├── model\_card.html             # Institutional model governance \& audit card

├── requirements.txt            # Dependency configurations

└── README.md                   # Technical documentation





\---



\## 6. Installation \& Local Deployment



\### Step 1: Clone Repository

```bash

git clone \[https://github.com/](https://github.com/)<your-username>/credit-risk-scoring-engine.git

cd credit-risk-scoring-engine

