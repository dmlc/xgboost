.. _release:

XGBoost Release Policy
=======================

Versioning Policy
---------------------------

Starting from XGBoost 1.0.0, each XGBoost release will be versioned as [MAJOR].[FEATURE].[MAINTENANCE]

* MAJOR: We gurantee the API compatibility across releases with the same major version number. We expect to have a 1+ years development period for a new MAJOR release version.
* FEATURE: We ship new features, improvements and bug fixes through feature releases. The cycle length of a feature is decided by the size of feature roadmap. The roadmap is decided right after the previous release.
* MAINTENANCE: Maintenance version only contains bug fixes. This type of release only occurs when we found significant correctness and/or performance bugs and barrier for users to upgrade to a new version of XGBoost smoothly.
