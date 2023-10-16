project(api-demo LANGUAGES C VERSION 0.0.1)
find_package(xgboost REQUIRED)

# xgboost is built as static libraries, all cxx dependencies need to be linked into the
# executable.
if(XGBOOST_BUILD_STATIC_LIB)
  enable_language(CXX)
  # find again for those  cxx libraries.
  find_package(xgboost REQUIRED)
endif()

add_executable(api-demo c-api-demo.c)
target_link_libraries(api-demo PRIVATE xgboost::xgboost)
