# Add multi-language support for tutorials

## Description
This PR addresses issue #11413 by adding R language examples alongside the existing Python examples in XGBoost tutorials. The goal is to improve the documentation by providing equivalent examples in both languages, making it more accessible to users who prefer working in R.

## Implementation
- Added language tabs using the sphinx-panels extension
- Created equivalent R code examples for the following tutorials:
  - [List of tutorials converted in this PR]
- Ensured consistent code style and functionality between Python and R examples
- Tested all R code examples to verify functionality

## Example
Before this PR, tutorials only provided Python examples:
```python
import xgboost as xgb
# Python-only example
```

After this PR, users can switch between Python and R examples:
```python
# Python tab
import xgboost as xgb
# Python example
```

```r
# R tab
library(xgboost)
# R equivalent example
```

## Related Issues
Fixes #11413 