import xgboost as xgb
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

n_samples = 1000
n_features = 10
X_dense = np.random.rand(n_samples, n_features)
y = np.random.randint(0, 2, n_samples)

# --- Case 1: CSR Matrix Input ---
X_csr_prep = X_dense.copy()
zero_indices = np.random.choice([True, False], size=X_csr_prep.shape, p=[0.2, 0.8])
X_csr_prep[zero_indices] = 0
X_csr = csr_matrix(X_csr_prep)

xgb_clf_csr = xgb.XGBClassifier(
    objective='binary:logistic',
    device='sycl',
    n_estimators=10,
    random_state=42
)

xgb_clf_csr.fit(X_csr, y)
y_pred_csr = xgb_clf_csr.predict(X_csr)

accuracy_csr = accuracy_score(y, y_pred_csr)
print(f"Accuracy with CSR data: {accuracy_csr:.4f}\n")

# --- Case 2: NumPy Array use 0.0 as missing value ---
X_numpy = X_csr.toarray()

xgb_clf_np = xgb.XGBClassifier(
    objective='binary:logistic',
    device='sycl',
    missing=0.0,
    n_estimators=10,
    random_state=42
)
xgb_clf_np.fit(X_numpy, y)
y_pred_np = xgb_clf_np.predict(X_numpy)

accuracy_np = accuracy_score(y, y_pred_np)
print(f"Accuracy with NumPy data: {accuracy_np:.4f}\n")
assert(accuracy_csr == accuracy_np)
