<div align="center">



\# Store Item Demand Forecasting Engine

\### Institutional-Grade Multi-Store Time-Series ML Pipeline for Retail Inventory Optimization



\[!\[Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=flat\&logo=python)](https://www.python.org/)

\[!\[Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E.svg?style=flat\&logo=scikit-learn)](https://scikit-learn.org/)

\[!\[Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg?style=flat\&logo=streamlit)](https://streamlit.io/)

\[!\[License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)



</div>



\---



\## 1. Executive Summary \& Business Impact



In multi-channel retail operations, inaccurate inventory forecasting directly creates dual capital inefficiencies:

\- \*\*Stockouts:\*\* Deplete on-shelf availability, causing immediate gross revenue loss and brand churn.

\- \*\*Overstocking:\*\* Drives warehouse congestion, holding cost decay, and forced end-of-season liquidation discounts.



This production-grade forecasting engine models daily transactional demand across \*\*50 retail stores\*\* and \*\*50 unique merchandise items\*\* over a 5-year chronological horizon (2019–2023). By replacing naive seasonal heuristics with a histogram-based gradient boosted regressor (`HistGradientBoostingRegressor`), this pipeline reduces forecast error by \*\*50.8%\*\*, enabling precise safety-stock allocation and inventory replenishment.



\---



\## 2. Core Benchmarks \& Performance Metrics



Models were benchmarked strictly out-of-sample on unseen Holdout Test data (H2 2023: July 1, 2023 to December 31, 2023) using domain-critical supply chain error formulas:



$$\\text{WAPE} = \\frac{\\sum\_{i=1}^n \\vert{}y\_i - \\hat{y}\_i\\vert{}}{\\sum\_{i=1}^n y\_i} \\times 100\\%$$



| Evaluation Model | Holdout WAPE (%) | MAE (Units) | RMSE (Units) | Outperformance vs. Baseline |

| :--- | :---: | :---: | :---: | :---: |

| \*\*Operational Heuristic (Lag-7 Naive Baseline)\*\* | 19.35% | 5.130 | 7.296 | \*Baseline\* |

| \*\*HistGradientBoosting Regressor (Trained Engine)\*\* | \*\*9.51%\*\* | \*\*2.520\*\* | \*\*3.158\*\* | \*\*+50.8% Error Reduction\*\* |



\---



\## 3. End-to-End Pipeline Architecture

Raw Daily Transactions (2019-2023)

│

▼

Temporal Integrity Audit (Zero-Gap Verification across 1,826 days)

│

▼

Feature Engineering Engine

├── Autoregressive Lags (t-1, t-7, t-14, t-30 per store/item)

├── Rolling Window Momentum (7-day \& 30-day Mean / Std shifted by t-1)

├── Calendar Signals (Day of Week, Quarter, Is-Weekend)

└── Cyclical Trigonometric Encodings (Sin/Cos of Week \& Month)

│

▼

Strict Chronological Out-of-Time Splitting

├── Train Horizon      : 2019-01-01 to 2022-12-31 (4 Years)

├── Validation Horizon : 2023-01-01 to 2023-06-30 (H1 2023)

└── Holdout Test       : 2023-07-01 to 2023-12-31 (H2 2023)

│

▼

HistGradientBoostingRegressor (Native Categorical Splits + Early Stopping)

│

▼

Model Serialization (artifacts/store\_demand\_forecaster.joblib)

│

▼

Interactive Streamlit Inventory Planning Dashboard



\## 4. Key Feature Importances (Permutation Impact on MAE)



1\. \*\*`rolling\_mean\_7` \& `lag\_7`:\*\* Dominant short-term weekly momentum anchors.

2\. \*\*`lag\_1`:\*\* Immediate previous-day run-rate capturing spontaneous purchasing spikes.

3\. \*\*`price` \& `promo`:\*\* Critical commercial elasticity drivers reflecting promotional lift.

4\. \*\*`sin\_dayofweek` / `cos\_dayofweek`:\*\* Strong intra-week cyclic rhythm capturing weekend demand surges.



\---



\## 5. Local Setup \& Execution Guide



\### Prerequisites

\- Python 3.10 or higher

\- Git



\### Installation

```bash

\# Clone the repository

git clone \[https://github.com/](https://github.com/)<your-username>/store-demand-forecasting-engine.git

cd store-demand-forecasting-engine



\# Create and activate virtual environment

python -m venv venv

source venv/bin/activate  # On Windows: venv\\Scripts\\activate



\# Install dependencies

pip install -r requirements.txt

Running the Application

Bash

streamlit run app.py

6\. Project Structure

├── artifacts/

│   └── store\_demand\_forecaster.joblib   # Serialized model \& metadata bundle

├── data/

│   └── sample\_input.csv                 # Minimal inference template

├── notebooks/

│   └── demand\_forecasting\_pipeline.ipynb # Complete Kaggle training notebook

├── app.py                               # Interactive Streamlit application

├── requirements.txt                     # Production dependencies

├── LICENSE                              # MIT License

└── README.md                            # Institutional documentation





