cmake_minimum_required(VERSION 3.12)
find_package(xgboost REQUIRED)
add_executable(api-demo c-api-demo.c)
target_link_libraries(api-demo xgboost::xgboost)
