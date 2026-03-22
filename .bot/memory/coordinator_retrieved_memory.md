# Retrieved Memory

## 1. agent_coordinator_plan
- repo: dmlc/xgboost
- adjusted_score: 0.2932
- timestamp: 2026-03-22T22:22:55.014139+00:00
# Coordinator Plan — dmlc/xgboost issue #11947

## Mission
Prepare a worker to implement the **smallest credible first slice** of the RFC: add a discoverable `xgboost.interpret` namespace with SHAP-oriented module functions, without trying to land the entire interpretability roadmap.

## Inputs reviewed
- `AGENTS.md`
- `.bot/retrieved_memory.md`
- `.bot/memory/coordinator_retrieved_memory.md`
- GitHub issue body and comments for `dmlc/xgboost#11947`
- Relevant source touchpoints in:
  - `python-package/xgboost/__init__.py`
  - `python-package/xgboost/core.py`
  - `python-package/xgboost/sklearn.py`
  - `tests/python/test_shap.py`

## What the issue actually needs
The RFC is broad, but the discussion strongly supports an incremental first step:
- add a **module-level** API under `xgboost.interpret`
- keep **computation separate from visualization**
- begin as a thin wrapper over existing:
  - `Booster.predict(..., pred_contribs=True)`
  - `Booster.predict(..., pred_interactions=True)`
- accept both `Booster` and sklearn-style `XGB*` models
- expose raw SHAP-friendly arrays now; defer larger result-class design

## Minimal patch to aim for
### In scope
1. New module: `python-package/xgboost/interpret.py`
2. New functions:
   - `shap_values(...)`
   - `shap_interactions(...)`
3. Public export so users can do:
   - `from xgboost import interpret`
4. Targeted Python tests for the new wrapper behavior

### Explicitly out of scope
Do **not** implement any of the following in the first pass:
- `topk_interactions`
- `partial_dependence`
- result object/dataclass hierarchy
- visualization helpers
- booster/sklearn convenience methods
- GroupSHAP / Owen values / Banzhaf work
- C++ changes
- docs beyond what is absolutely required for imports/tests

## Design decisions to preserve
### 1) Module-level functions, not new model methods
The issue discussion leans toward keeping interpretation functionality separated from the already-large model APIs. The worker should not add `shap_values()` methods onto booster/sklearn classes in this patch.

### 2) `shap_values` should not return the bias term inline by default
Existing `pred_contribs=True` returns `n_features + 1`, with the final column as bias. The issue discussion explicitly notes that returning that full matrix by default is awkward for downstream SHAP plotting.

**Desired first-pass behavior:**
- default return: feature-only values, shape `(..., n_features)`
- `return_bias=True`: return `(values, bias)` where `bias` is the separated last contribution axis

### 3) `shap_interactions` should remove the bias row/column
Existing `pred_interactions=True` returns the bias in the final row/column. The wrapper should return the feature-only interaction tensor, shape `(..., n_features, n_features)`.

### 4) Preserve sklearn early-stopping iteration behavior
`XGBModel.predict()` uses `_get_iteration_range(None)` to honor `best_iteration`. The wrapper should mirror that behavior when the passed model exposes `_get_i

## 2. issue_summary
- repo: t7r0n/OSS_3
- adjusted_score: 0.1908
- timestamp: 2026-03-21T23:13:39.941080+00:00
Issue #4: Supervisor non-dry-run README patch
Append one line to README.md: Supervisor live-mode sentence.

## 3. repo_policy
- repo: dmlc/xgboost
- adjusted_score: 0.1040
- timestamp: 2026-03-22T22:25:08.433510+00:00
LICENSE:                                  Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control sy

## 4. repo_policy
- repo: dmlc/xgboost
- adjusted_score: 0.0795
- timestamp: 2026-03-22T22:25:08.387465+00:00
README.md: <img src="https://xgboost.ai/images/logo/xgboost-logo-trimmed.png" width=200/> eXtreme Gradient Boosting
===========

[![XGBoost-CI](https://github.com/dmlc/xgboost/workflows/XGBoost%20CI/badge.svg?branch=master)](https://github.com/dmlc/xgboost/actions)
[![Documentation Status](https://readthedocs.org/projects/xgboost/badge/?version=latest)](https://xgboost.readthedocs.org)
[![GitHub license](https://dmlc.github.io/img/apache2.svg)](./LICENSE)
[![CRAN Status Badge](https://www.r-pkg.org/badges/version/xgboost)](https://cran.r-project.org/web/packages/xgboost)
[![PyPI version](https://badge.fury.io/py/xgboost.svg)](https://pypi.python.org/pypi/xgboost/)
[![Conda version](https://img.shields.io/conda/vn/conda-forge/py-xgboost.svg)](https://anaconda.org/conda-forge/py-xgboost)
[![Optuna](https://img.shields.io/badge/Optuna-integrated-blue)](https://optuna.org)
[![Twitter](https://img.shields.io/badge/@XGBoostProject--_.svg?style=social&logo=twitter)](https://twitter.com/XGBoostProject)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/dmlc/xgboost/badge)](https://api.securityscorecards.dev/projects/github.com/dmlc/xgboost)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/comet-examples/blob/master/integrations/model-training/xgboost/notebooks/how_to_use_comet_with_xgboost_tutorial.ipynb)

[Community](https://xgboost.ai/community) |
[Documentation](https://xgboost.readthedocs.org) |
[Resources](demo/README.md) |
[Contributors](CONTRIBUTORS.md) |
[Release Notes](https://xgboost.readthedocs.io/en/latest/changes/index.html)

XGBoost is an optimized distributed gradient boosting library designed to be highly ***efficient***, ***flexible*** and ***portable***.
It implements machine learning algorithms under the [Gradient Boosting](https://en.wikipedia.org/wiki/Gradient_boosting) framework.
XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that solve many data science problems in a fast and accurate way.
The same code runs on major distributed environment (Kubernetes, Hadoop, SGE, Dask, Spark, PySpark) and can solve problems beyond billions of examples.

License
-------
© Contributors, 2021. Licensed under an [Apache-2](https://github.com/dmlc/xgboost/blob/master/LICENSE) license.

Contribute to XGBoost
---------------------
XGBoost has been developed and used by a group of active community members. Your help is very valuable to make the package better for everyone.
Checkout the [Community Page](https://xgboost.ai/community).

Reference
---------
- Tianqi Chen and Carlos Guestrin. [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754). In 22nd SIGKDD Conference on Knowledge Discovery and Data Mining, 2016
- XGBoost originates from research project at University of Washington.

Sponsors
--------
Become a sponsor and get a logo here. See details at [Sponsoring the XGBoost Project](https://xgboost.ai/sponsors). The 

## 5. issue_summary
- repo: t7r0n/OSS_3
- adjusted_score: -0.0400
- timestamp: 2026-03-21T22:40:33.842979+00:00
Issue #3: Append explicit README sentence
Append exactly one line to README.md: Supervisor validation sentence added by Codex.

## 6. repo_policy
- repo: t7r0n/OSS_2
- adjusted_score: -0.1973
- timestamp: 2026-03-21T22:27:16.186327+00:00
README.md: # OSS_2
OpenClaw/Codex smoke validation repo

This repository was validated in openclaw/codex stack smoke testing on March 21, 2026.


## 7. repo_policy
- repo: t7r0n/OSS_3
- adjusted_score: -0.2608
- timestamp: 2026-03-21T22:33:53.067481+00:00
README.md: # OSS_3
Python supervisor smoke validation repo


## 8. issue_summary
- repo: t7r0n/OSS_3
- adjusted_score: -0.3449
- timestamp: 2026-03-21T22:36:50.724063+00:00
Issue #2: Add second supervisor sentence to README
Append another one-line sentence in README to validate supervisor + codex execution.
