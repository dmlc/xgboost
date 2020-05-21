find_package(JNI REQUIRED)

add_library(xgboost4j SHARED
  ${PROJECT_SOURCE_DIR}/jvm-packages/xgboost4j/src/native/xgboost4j.cpp
  ${XGBOOST_OBJ_SOURCES})
target_include_directories(xgboost4j
  PRIVATE
  ${JNI_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/jvm-packages/xgboost4j/src/native
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/dmlc-core/include
  ${PROJECT_SOURCE_DIR}/rabit/include)

set_output_directory(xgboost4j ${PROJECT_SOURCE_DIR}/lib)
set_target_properties(
  xgboost4j PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON)
target_link_libraries(xgboost4j
  PRIVATE
  ${LINKED_LIBRARIES_PRIVATE}
  ${JAVA_JVM_LIBRARY})
