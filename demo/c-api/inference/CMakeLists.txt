cmake_minimum_required(VERSION 3.13)
project(inference-demo LANGUAGES C VERSION 0.0.1)
find_package(xgboost REQUIRED)

# xgboost is built as static libraries, all cxx dependencies need to be linked into the
# executable.
if (XGBOOST_BUILD_STATIC_LIB)
  enable_language(CXX)
  # find again for those  cxx libraries.
  find_package(xgboost REQUIRED)
endif(XGBOOST_BUILD_STATIC_LIB)

add_executable(inference-demo inference.c)
target_link_libraries(inference-demo PRIVATE xgboost::xgboost)
