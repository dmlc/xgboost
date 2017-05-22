#------------------------------------------------------------------------
# xgboost: minumum dependency configuration with Parallelization.
# This configuration is standard but cannot run distributed computing.
#
# see config.mk for template.
#------------------------------------------------------------------------

# Whether enable openmp support, needed for multi-threading.
USE_OPENMP = 1

# whether use HDFS support during compile
USE_HDFS = 0

# whether use AWS S3 support during compile
USE_S3 = 0

# whether use Azure blob support during compile
USE_AZURE = 0

# Rabit library version,
# - librabit.a Normal distributed version.
# - librabit_empty.a Non distributed mock version,
LIB_RABIT = librabit_empty.a
