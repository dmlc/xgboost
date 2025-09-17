cmake_minimum_required(VERSION 3.18)
project(external-memory-demo LANGUAGES C VERSION 0.0.1)

find_package(xgboost REQUIRED)

add_executable(external-memory-demo external_memory.c)
target_link_libraries(external-memory-demo PRIVATE xgboost::xgboost)
