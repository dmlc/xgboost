# README Updates for Federated Learning

## Current Status
✅ README.md already mentions XGBoost works with federated learning in line 25:
> "The same code runs on major distributed environment (Kubernetes, Hadoop, SGE, Dask, Spark, PySpark) and can solve problems beyond billions of examples."

However, federated learning should be more explicitly highlighted as a **key feature**.

---

## Recommended Updates

### 1. Update Feature Highlights Section

**Location:** After line 22-25 (the paragraph about parallel tree boosting)

**Add a new bullet point:**

```markdown
## Key Features

* **Distributed and Federated Learning**:
  - Seamlessly run on major distributed environments (Kubernetes, Hadoop, SGE, Dask, Spark, PySpark)
  - Built-in support for privacy-preserving federated learning across multiple parties
  - GPU-accelerated federated training with secure aggregation
```

---

### 2. Create a "Federated Learning" Section

**Location:** After the main "XGBoost is..." introduction paragraph, before "License" section

**Add:**

```markdown
## Federated Learning

XGBoost supports **privacy-preserving federated learning**, enabling multiple parties to collaboratively train models without sharing raw data:

- **Horizontal Federated Learning**: Data split by samples (different records, same features)
- **Vertical Federated Learning**: Data split by features (different features, same records)
- **Secure Aggregation**: Encrypted gradient and histogram synchronization
- **GPU Acceleration**: Full CUDA support for federated training

```python
# Example: Federated learning setup
import xgboost as xgb
import xgboost.federated

# Start federated server
xgboost.federated.run_federated_server(n_workers=3, port=9091)

# Worker-side training (on each party)
with xgb.collective.CommunicatorContext(
    dmlc_communicator='federated',
    federated_server_address='localhost:9091',
    federated_world_size=3,
    federated_rank=0
):
    dtrain = xgb.DMatrix('local_data.txt')
    bst = xgb.train({'tree_method': 'hist'}, dtrain, num_boost_round=100)
```

Learn more in the [Federated Learning Tutorial](https://xgboost.readthedocs.io/en/latest/tutorials/federated_learning.html).

```

---

### 3. Update "Contribute to XGBoost" Section

**No changes needed** - This section already points to the Community Page which can be updated separately.

---

### 4. Alternative: Minimal Update (If You Want to Keep README Short)

If you prefer to keep the README concise, just update the existing paragraph:

**Replace line 25:**

```markdown
The same code runs on major distributed environment (Kubernetes, Hadoop, SGE, Dask, Spark, PySpark) and can solve problems beyond billions of examples.
```

**With:**

```markdown
The same code runs on major distributed environments (Kubernetes, Hadoop, SGE, Dask, Spark, PySpark) and supports **privacy-preserving federated learning**. XGBoost can solve problems beyond billions of examples while preserving data privacy across multiple parties.
```

And add a link to the documentation:

```markdown
For federated learning, see the [Federated Learning Tutorial](https://xgboost.readthedocs.io/en/latest/tutorials/federated_learning.html).
```

---

## Complete Updated README Section

Here's what the updated intro section could look like:

```markdown
<img src="https://xgboost.ai/images/logo/xgboost-logo-trimmed.png" width=200/> eXtreme Gradient Boosting
===========

[![Build Status](https://badge.buildkite.com/aca47f40a32735c00a8550540c5eeff6a4c1d246a580cae9b0.svg?branch=master)](https://buildkite.com/xgboost/xgboost-ci)
<!-- ...other badges... -->

[Community](https://xgboost.ai/community) |
[Documentation](https://xgboost.readthedocs.org) |
[Resources](demo/README.md) |
[Contributors](CONTRIBUTORS.md) |
[Release Notes](https://xgboost.readthedocs.io/en/latest/changes/index.html) |
[Federated Learning](https://xgboost.readthedocs.io/en/latest/tutorials/federated_learning.html)

XGBoost is an optimized distributed gradient boosting library designed to be highly ***efficient***, ***flexible*** and ***portable***.
It implements machine learning algorithms under the [Gradient Boosting](https://en.wikipedia.org/wiki/Gradient_boosting) framework.
XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that solve many data science problems in a fast and accurate way.
The same code runs on major distributed environments (Kubernetes, Hadoop, SGE, Dask, Spark, PySpark) and supports **privacy-preserving federated learning** across multiple parties.
XGBoost can solve problems beyond billions of examples while protecting data privacy.

## Key Features

* **Speed and Performance**: Highly optimized, efficient, and scalable
* **Flexibility**: Supports custom objectives and evaluation metrics
* **Portability**: Runs on Windows, Linux, MacOS with multiple language bindings (Python, R, Java, Scala, Julia, etc.)
* **Distributed Computing**: Native support for distributed training on Kubernetes, Hadoop, Dask, Spark, and more
* **Federated Learning**: Privacy-preserving training across multiple parties without sharing raw data
* **GPU Acceleration**: CUDA-accelerated training including federated learning scenarios
* **Model Interpretability**: Built-in feature importance, SHAP value support

For federated learning capabilities, see the [Federated Learning Tutorial](https://xgboost.readthedocs.io/en/latest/tutorials/federated_learning.html).
```

---

## What NOT to Change

❌ **Don't remove or modify:**
- License section
- Sponsor information
- Existing badges
- Contribution guidelines link
- Reference/citation information

✅ **Only add/update:**
- Feature descriptions
- Links to new documentation
- Example code (optional)

---

## Verification

After updating README.md:

1. **Check Markdown rendering:**
   ```bash
   # Preview on GitHub (push to your fork first)
   # Or use a local markdown previewer
   ```

2. **Verify all links work:**
   ```bash
   # Check that the federated learning tutorial link resolves
   # URL: https://xgboost.readthedocs.io/en/latest/tutorials/federated_learning.html
   ```

3. **Test code examples (if added):**
   ```bash
   # Make sure any code snippets in README are syntactically correct
   python -m py_compile <extracted_code.py>
   ```

4. **Check formatting:**
   - Markdown tables render correctly
   - Code blocks have proper syntax highlighting (```python)
   - Links are properly formatted [text](url)
   - Badges still display correctly

---

## Git Commit Message

When committing README updates:

```
[doc] Update README with federated learning features

- Highlight federated learning as a key feature
- Add link to federated learning tutorial
- Update feature list to include privacy-preserving training
- Optional: Add federated learning code example

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

---

## Summary

### Required Changes:
1. ✅ Update the main intro paragraph to mention federated learning
2. ✅ Add federated learning to key features list
3. ✅ Add link to federated learning tutorial in header links

### Optional Changes:
4. ⬜ Add a dedicated "Federated Learning" section with code example
5. ⬜ Update badges/shields if there's a federated learning demo

### Minimal Changes (if you want to keep README short):
- Just update line 25 to mention federated learning
- Add tutorial link to the header navigation

**Recommendation:** Go with the **minimal update** approach since README.md should stay concise. Detailed information belongs in the documentation.
