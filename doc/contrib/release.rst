.. _release:

XGBoost Release Policy
=======================

Versioning Policy
---------------------------

Starting from XGBoost 1.0.0, each XGBoost release will be versioned as [MAJOR].[FEATURE].[MAINTENANCE]

* MAJOR: We guarantee the API compatibility across releases with the same major version number. We expect to have a 1+ years development period for a new MAJOR release version.
* FEATURE: We ship new features, improvements and bug fixes through feature releases. The cycle length of a feature is decided by the size of feature roadmap. The roadmap is decided right after the previous release.
* MAINTENANCE: Maintenance version only contains bug fixes. This type of release only occurs when we found significant correctness and/or performance bugs and barrier for users to upgrade to a new version of XGBoost smoothly.


Making a Release
-----------------

1. Create an issue for the release, noting the estimated date and expected features or major fixes, pin that issue.
2. Bump release version.

   1. Modify ``CMakeLists.txt`` in source tree and ``cmake/Python_version.in`` if needed, run CMake.

   2. Modify ``DESCRIPTION`` in R-package.

   3. Run ``change_version.sh`` in ``jvm-packages/dev``

3. Commit the change, create a PR on GitHub on release branch.  Port the bumped version to default branch, optionally with the postfix ``SNAPSHOT``.
4. Create a tag on release branch, either on GitHub or locally.
5. Make a release on GitHub tag page, which might be done with previous step if the tag is created on GitHub.
6. Submit pip, CRAN, and Maven packages.

   + The pip package is maintained by `Hyunsu Cho <https://github.com/hcho3>`__ and `Jiaming Yuan <https://github.com/trivialfis>`__.  There's a helper script for downloading pre-built wheels and R packages ``xgboost/dev/release-pypi-r.py`` along with simple instructions for using ``twine``.

   + The CRAN package is maintained by `Tong He <https://github.com/hetong007>`_ and `Jiaming Yuan <https://github.com/trivialfis>`__.

     Before submitting a release, one should test the package on `R-hub <https://builder.r-hub.io/>`__ and `win-builder <https://win-builder.r-project.org/>`__ first.  Please note that the R-hub Windows instance doesn't have the exact same environment as the one hosted on win-builder.

   + The Maven package is maintained by `Nan Zhu <https://github.com/CodingCat>`_ and `Hyunsu Cho <https://github.com/hcho3>`_.
