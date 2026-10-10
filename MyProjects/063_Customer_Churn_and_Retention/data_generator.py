import numpy as np
import pandas as pd
import os

np.random.seed(42)
n_samples = 2500

# 1. Customer attributes simulate karein
customer_ids = [f"CUST-{1000 + i}" for i in range(n_samples)]
tenure_months = np.random.randint(1, 48, size=n_samples)
contract_type = np.random.choice(['Month-to-Month', 'One-Year', 'Two-Year'], size=n_samples, p=[0.55, 0.25, 0.20])
monthly_charges = np.random.uniform(20.0, 150.0, size=n_samples).round(2)
support_tickets = np.random.poisson(lam=1.8, size=n_samples)
usage_drop_pct = np.random.uniform(-0.5, 0.6, size=n_samples).round(2)
payment_failures = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.75, 0.15, 0.07, 0.03])

# 2. Churn probability calculation (sabhi jagah * operator verify kiya gaya hai)
log_odds = (
    - (0.05 * tenure_months)
    + (0.015 * monthly_charges)
    + (0.45 * support_tickets)
    - (2.2 * usage_drop_pct)
    + (0.8 * payment_failures)
    + np.where(contract_type == 'Month-to-Month', 0.8, -0.6)
    - 1.2
)

prob_churn = 1 / (1 + np.exp(-log_odds))
churn = (np.random.rand(n_samples) < prob_churn).astype(int)

# 3. CSV save karein
df = pd.DataFrame({
    'customer_id': customer_ids,
    'tenure_months': tenure_months,
    'contract_type': contract_type,
    'monthly_charges': monthly_charges,
    'support_tickets': support_tickets,
    'usage_drop_pct': usage_drop_pct,
    'payment_failures': payment_failures,
    'churn': churn
})

os.makedirs('data', exist_ok=True)
csv_path = os.path.join('data', 'saas_customer_churn.csv')
df.to_csv(csv_path, index=False)
print(f"Dataset generated: {csv_path} ({len(df)} rows, Churn rate: {df['churn'].mean():.2%})")