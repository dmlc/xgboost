# XGBoost Multi-Language Tutorials

This project aims to make all XGBoost tutorials multi-language, by adding R code examples alongside the existing Python examples. This will allow users to choose their preferred language when reading the documentation.

## Background

XGBoost is a popular machine learning library that supports multiple programming languages, including Python, R, Java, and more. However, most of the current tutorials only provide examples in Python. This project addresses [Issue #11413](https://github.com/dmlc/xgboost/issues/11413) by making all tutorials multi-language.

## How to Contribute

If you'd like to contribute to this effort, follow these steps:

1. Check the [PLAN.md](PLAN.md) file to see which tutorials still need R examples.
2. Choose a tutorial from the list that hasn't been converted yet.
3. Fork the repository and create a new branch for your changes.
4. Follow the template in [example_language_tabs.rst](doc/example_language_tabs.rst) to add language tabs to the tutorial.
5. Write R code examples that are equivalent to the existing Python examples.
6. Test your R code to ensure it works correctly.
7. Submit a pull request with your changes.

## Guidelines for R Code Examples

When writing R code examples:

1. Use the same datasets as the Python examples when possible, or equivalent datasets that are available in R.
2. Follow R coding conventions and style guidelines.
3. Include comments to explain the code, similar to the Python examples.
4. Keep the examples concise and focused on the same concepts as the Python examples.
5. Test all code before submitting.

## Example of Language Tabs

Here's how the language tabs look in the documentation:

```rst
Training a Boosted Tree with XGBoost
====================================

.. tabbed:: Python

    .. code-block:: python

        import xgboost as xgb
        # Python example code here

.. tabbed:: R

    .. code-block:: r

        library(xgboost)
        # R equivalent code here
```

## Progress Tracking

We'll track progress in the [PLAN.md](PLAN.md) file, marking each tutorial as it's completed. The goal is to convert all applicable tutorials to have both Python and R examples.

## Resources

- [XGBoost Python Documentation](https://xgboost.readthedocs.io/en/latest/python/index.html)
- [XGBoost R Documentation](https://xgboost.readthedocs.io/en/latest/R-package/index.html)
- [sphinx-panels Documentation](https://sphinx-panels.readthedocs.io/en/latest/) (for language tabs) 