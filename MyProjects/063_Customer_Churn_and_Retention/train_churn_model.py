import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

print("1. Loading dataset...")
data_path = os.path.join('data', 'saas_customer_churn.csv')
if not os.path.exists(data_path):
    raise FileNotFoundError("data/saas_customer_churn.csv nahi mili! Pehle data_generator.py run karein.")

df = pd.read_csv(data_path)

# Features aur Target alag karein
feature_cols = [
    'tenure_months', 
    'contract_type', 
    'monthly_charges', 
    'support_tickets', 
    'usage_drop_pct', 
    'payment_failures'
]
target_col = 'churn'

X = df[feature_cols]
y = df[target_col]

# Train-Test Split (Stratified taaki churn ratio barabar rahe)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Dataset split: Train={len(X_train)} samples, Test={len(X_test)} samples")
print(f"Overall Churn Rate: {y.mean():.2%}")

# 2. Pipeline Architecture
numeric_features = ['tenure_months', 'monthly_charges', 'support_tickets', 'usage_drop_pct', 'payment_failures']
categorical_features = ['contract_type']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
    ]
)

# HistGradientBoostingClassifier
model = HistGradientBoostingClassifier(
    max_iter=200,
    learning_rate=0.08,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    random_state=42
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# 3. Model Training
print("2. Training Churn Prediction Pipeline...")
pipeline.fit(X_train, y_train)

# 4. Evaluation
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_prob)
print("\n--- Model Performance Summary ---")
print(f"ROC-AUC Score: {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

# 5. Artifacts Packaging
os.makedirs('artifacts', exist_ok=True)
artifact_path = os.path.join('artifacts', 'churn_model_bundle.joblib')

artifact_payload = {
    'pipeline': pipeline,
    'feature_cols': feature_cols,
    'numeric_features': numeric_features,
    'categorical_features': categorical_features,
    'metrics': {
        'roc_auc': round(roc_auc, 4),
        'test_samples': len(X_test),
        'churn_rate': round(float(y.mean()), 4)
    }
}

joblib.dump(artifact_payload, artifact_path)
print(f"\n[SUCCESS] Model artifact bundle saved at: {artifact_path}")