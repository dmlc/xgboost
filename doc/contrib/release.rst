.. _release:

XGBoost Release Policy
=======================

Versioning Policy
---------------------------

Starting from XGBoost 1.0.0, each XGBoost release will be versioned as [MAJOR].[FEATURE].[MAINTENANCE]

* MAJOR: We gurantee the API compatibility across releases with the same major version umber. We expect to have a 1+ years development period for a new MAJOR release version.
* FEATURE: We ship new features, improvements and bug fixes through feature releases. The cycle length of a feature is decided by the size of feature roadmap. The roadmap is decided right after the previous release. 


MAINTENANCE: Maintenance releases will occur more frequently and depend on specific patches introduced (e.g. bug fixes) and their urgency. In general these releases are designed to patch bugs. However, higher level libraries may introduce small features, such as a new algorithm, provided they are entirely additive and isolated from existing code paths. Spark core may not introduce any features.
