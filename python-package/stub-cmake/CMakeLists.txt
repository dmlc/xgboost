cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Empty CMake project for XGBoost stub packages (xgboost-cu12, xgboost-cu13).
# Stubs are metadata-only redirects to the canonical `xgboost` distribution;
# they ship no Python modules and no shared library.
project(xgboost_stub LANGUAGES NONE)
