import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1. ARCHITECTURAL SETUP
districts = ['A', 'B', 'C']
features = [
    'attendance', 'avg_score', 'consistency_index', 
    'time_on_task_min', 'lms_interactions', 
    'weekend_ratio', 'late_night_freq'
]

print("--- PROJECT ATHENAL: FEDERATED LEARNING PROTOCOL (FedAvg) ---")

# 2. SERVER VALIDATION SET

try:
    global_test = pd.concat([pd.read_csv(f'district_{x}.csv') for x in districts]).sample(frac=0.2, random_state=42)
except FileNotFoundError:
    print("❌ ERROR: District CSV files not found. run - athenal_xgboost.py first")
    exit()

X_test = global_test[features]
y_test = global_test['label']

# Global Scaler (Simulating standardized preprocessing protocol across districts)
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)

# 3. DISTRIBUTED LOCAL TRAINING (The 'Silos')
local_weights = []
local_intercepts = []

print(f"\n[Round 1] Initiating Local Training on {len(districts)} District Nodes...")

for d in districts:
    # District trains ONLY on its local silo
    df = pd.read_csv(f'district_{d}.csv')
    X_local = df[features]
    y_local = df['label']
    
    # Local independent scaling
    local_scaler = StandardScaler()
    X_local_scaled = local_scaler.fit_transform(X_local)
    
    # We use Logistic Regression coefficients to demonstrate Weight Aggregation
    model = LogisticRegression(max_iter=1000)
    model.fit(X_local_scaled, y_local)
    
    # We extract ONLY the mathematical coefficients (weights)
    # The student records remain inside the District Node.

    #point 3 - most important line - mathematical coefficients.
    local_weights.append(model.coef_)
    local_intercepts.append(model.intercept_)
    
    local_acc = accuracy_score(y_local, model.predict(X_local_scaled))
    print(f"  ✓ Node {d}: Training Complete. Local Acc: {local_acc:.2%}. weights.bin transmitted.")

# 4. SECURE AGGREGATION (The Central Server)
print("\n[Server] Performing Federated Averaging (FedAvg)...")

#point 4 - Fed Avg algorithm
global_weights = np.mean(local_weights, axis=0)
global_intercept = np.mean(local_intercepts, axis=0)

# 5. GLOBAL MODEL DEPLOYMENT
global_model = LogisticRegression()
global_model.coef_ = global_weights
global_model.intercept_ = global_intercept
global_model.classes_ = np.array([0, 1])

# 6. FINAL ARCHITECTURAL VERIFICATION
y_pred = global_model.predict(X_test_scaled)
federated_acc = accuracy_score(y_test, y_pred)

print("-" * 60)
print(f"FINAL FEDERATED MODEL ACCURACY: {federated_acc:.2%}")
print("-" * 60)
print("\n--- FEDERATED VALIDATION COMPLETE ---")