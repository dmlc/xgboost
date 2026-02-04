# Agents Guide for XGBoost

This document provides guidance for AI coding agents and assistants working with the XGBoost repository.

## About XGBoost

XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable. It implements machine learning algorithms under the Gradient Boosting framework and provides parallel tree boosting (also known as GBDT, GBM) that solves many data science problems in a fast and accurate way.

## Repository Structure

The XGBoost repository is organized as follows:

- **`src/`**: Core C++ implementation of XGBoost algorithms
- **`include/`**: C++ header files and public APIs
- **`python-package/`**: Python package implementation
- **`R-package/`**: R package implementation
- **`jvm-packages/`**: Java/Scala packages (JVM-based implementations)
- **`tests/`**: Test suites for various components
- **`doc/`**: Documentation source files (Sphinx-based)
- **`demo/`**: Example code and tutorials
- **`plugin/`**: Plugin implementations for various frameworks
- **`cmake/`**: CMake build configuration files

## Build System

XGBoost uses **CMake** as its primary build system. Key build files:
- `CMakeLists.txt`: Main build configuration
- `cmake/`: Additional CMake modules and configurations

### Building XGBoost

```bash
# Basic build
mkdir build
cd build
cmake ..
make -j4

# Build with GPU support
cmake .. -DUSE_CUDA=ON
make -j4

# Build with tests
cmake .. -DGOOGLE_TEST=ON
make -j4
```

## Testing

XGBoost has comprehensive test coverage across multiple languages:

### C++ Tests
```bash
cd build
ctest
```

### Python Tests
```bash
cd python-package
pytest tests/python
```

### R Tests
```bash
cd R-package
R CMD check xgboost_*.tar.gz
```

## Code Style and Linting

**Use pre-commit for all formatting and linting** - this is the recommended and primary way to ensure code quality:

```bash
# Install pre-commit hooks (one-time setup)
python -m pip install pre-commit
pre-commit install

# Format and lint staged files
pre-commit run

# Format and lint all files
pre-commit run --all-files

# Format and lint specific files
pre-commit run --files path/to/file1.cc path/to/file2.py
```

The pre-commit configuration (`.pre-commit-config.yaml`) includes:

### C++ Formatting and Linting
- **clang-format**: Automatic code formatting (config: `.clang-format`)
  - Follows Google C++ Style Guide with customizations
  - Formats `.cc`, `.c`, `.cpp`, `.h`, `.cu`, `.hpp` files
- **cpplint**: Style checking and linting
- **clang-tidy**: Static analysis (config: `.clang-tidy`)

### Python Formatting and Linting
- **black**: Automatic code formatting
- **isort**: Import sorting
- **pylint**: Style checking and linting (config: `python-package/pyproject.toml`)
- **mypy**: Type checking (run separately: `cd python-package && mypy xgboost`)

### CMake Linting
- **cmakelint**: Checks CMake files for style issues

## Key Development Areas

### 1. Core Algorithms (C++)
Located in `src/`, includes:
- Tree construction algorithms
- Gradient boosting implementations
- Objective functions and metrics
- GPU acceleration code

### 2. Python Interface
Located in `python-package/`, includes:
- Scikit-learn compatible API
- Native XGBoost API
- Plotting and visualization utilities
- Dask and Spark integrations

### 3. R Interface
Located in `R-package/`, provides:
- R-native API
- Integration with R ecosystem
- R-specific documentation

### 4. JVM Packages
Located in `jvm-packages/`, includes:
- Java API
- Scala API
- Spark integration

## Common Development Tasks

### Adding a New Feature
1. Implement core logic in C++ (`src/`)
2. Add appropriate headers (`include/`)
3. Expose through language bindings (Python/R/JVM)
4. Add comprehensive tests
5. Update documentation (`doc/`)
6. Add demo/example if applicable

### Fixing a Bug
1. Write a failing test that reproduces the bug
2. Fix the bug in the appropriate layer (C++/Python/R/JVM)
3. Ensure all tests pass
4. Update documentation if behavior changes

### Performance Optimization
1. Profile the code to identify bottlenecks
2. Implement optimizations in C++ core
3. Add benchmarks to track performance
4. Validate correctness with existing tests

## Documentation

XGBoost uses **Sphinx** for documentation:
- Source files: `doc/`
- Configuration: `doc/conf.py`
- Build documentation: `cd doc && make html`

Documentation is hosted at: https://xgboost.readthedocs.io

## Continuous Integration

XGBoost uses GitHub Actions for CI/CD:
- Workflows defined in `.github/workflows/`
- Tests run on multiple platforms (Linux, macOS, Windows)
- GPU tests run on NVIDIA infrastructure

## Dependencies

### Core C++ Dependencies
- Minimal dependencies for core library
- Optional: CUDA for GPU support
- Optional: OpenMP for CPU parallelization

### Python Dependencies
- NumPy
- SciPy
- scikit-learn (for compatibility)
- See `python-package/setup.py` for full list

### Build Dependencies
- CMake >= 3.13
- C++14 compatible compiler (GCC 5+, Clang 3.4+, MSVC 2017+)

## Important Guidelines for Agents

### 1. Maintain Backward Compatibility
XGBoost is widely used in production. Avoid breaking changes to public APIs unless absolutely necessary and properly deprecated.

### 2. Performance is Critical
XGBoost is a performance-critical library. Always consider performance implications of changes, especially in hot paths.

### 3. Cross-Platform Support
Code must work on Linux, macOS, and Windows. Test on multiple platforms when possible.

### 4. Multi-Language Support
Changes to core C++ often require corresponding updates to Python, R, and JVM bindings.

### 5. Comprehensive Testing
Add tests for new features and bug fixes. Maintain high test coverage.

### 6. Documentation
Update documentation for user-facing changes. Include docstrings for new APIs.

### 7. Code Review
All changes go through code review. Follow reviewer feedback and maintain respectful communication.

## Getting Help

- **Documentation**: https://xgboost.readthedocs.io
- **Community Forum**: https://discuss.xgboost.ai
- **GitHub Issues**: https://github.com/dmlc/xgboost/issues
- **Contributing Guide**: See CONTRIBUTORS.md

## Security

Report security vulnerabilities to security@xgboost-ci.net. See SECURITY.md for details.

## License

XGBoost is licensed under the Apache License 2.0. All contributions must comply with this license.

## Common Patterns and Idioms

### Error Handling in C++
```cpp
#include <dmlc/logging.h>
CHECK(condition) << "Error message";
CHECK_EQ(a, b) << "Values must be equal";
LOG(INFO) << "Informational message";
```

### Python API Design
```python
# Follow scikit-learn conventions
def fit(self, X, y=None, sample_weight=None, **kwargs):
    """Fit the model.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data
    y : array-like of shape (n_samples,)
        Target values
    """
    pass
```

### CMake Best Practices
```cmake
# Use target-based CMake
target_include_directories(objxgboost PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(xgboost PRIVATE objxgboost)
```

## Version Management

- Version defined in: `CMakeLists.txt`, `python-package/xgboost/VERSION`, `R-package/DESCRIPTION`
- Keep versions synchronized across all packages
- Follow semantic versioning (MAJOR.MINOR.PATCH)

## Release Process

1. Update version numbers in all relevant files
2. Update NEWS.md with release notes
3. Create release tag
4. Build and test release artifacts
5. Publish to PyPI, CRAN, Maven Central
6. Update documentation

## Advanced Topics

### GPU Programming
- CUDA code in `src/tree/gpu_hist/`
- Use of Thrust library for GPU algorithms
- Device memory management patterns

### Distributed Computing
- Dask integration: `python-package/xgboost/dask.py`
- Spark integration: `jvm-packages/xgboost4j-spark/`
- Communication patterns for distributed training

### Custom Objectives and Metrics
- Custom objectives: `src/objective/`
- Custom metrics: `src/metric/`
- User-defined functions in Python/R

This guide should help AI agents understand the XGBoost codebase structure, development workflows, and best practices for contributing to the project.
