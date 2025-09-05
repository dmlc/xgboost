import xgboost as xgb
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

n_samples = int(1e7)
n_features = int(1e3)
X_dense = np.random.rand(n_samples, n_features)
y = np.random.randint(0, 2, n_samples)

sparsity = 1e-5
zero_indices = np.random.choice([True, False], size=X_dense.shape, p=[1-sparsity, sparsity])
X_dense[zero_indices] = 0

# --- Case 1: CSR Matrix Input ---
X_csr = csr_matrix(X_dense)

xgb_clf_csr = xgb.XGBClassifier(
    objective='binary:logistic',
    device='sycl',
    n_estimators=1,
    random_state=42
)

xgb_clf_csr.fit(X_csr, y)
y_pred_csr = xgb_clf_csr.predict(X_csr)

accuracy_csr = accuracy_score(y, y_pred_csr)
print(f"Accuracy with CSR data: {accuracy_csr:.4f}\n")

# --- Case 2: Dense Matrix Input ---
xgb_clf_np = xgb.XGBClassifier(
    objective='binary:logistic',
    device='sycl',
    n_estimators=1,
    random_state=42
)
try:
  xgb_clf_np.fit(X_dense, y)
  y_pred_np = xgb_clf_np.predict(X_dense)

  accuracy_np = accuracy_score(y, y_pred_np)
  print(f"Accuracy with NumPy data: {accuracy_np:.4f}\n")
except:
  print("Out of memory on device")
