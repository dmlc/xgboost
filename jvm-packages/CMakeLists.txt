find_package(JNI REQUIRED)

add_library(xgboost4j SHARED
  ${PROJECT_SOURCE_DIR}/jvm-packages/xgboost4j/src/native/xgboost4j.cpp)
if (ENABLE_ALL_WARNINGS)
  target_compile_options(xgboost4j PUBLIC -Wall -Wextra)
endif (ENABLE_ALL_WARNINGS)
target_link_libraries(xgboost4j PRIVATE objxgboost)
target_include_directories(xgboost4j
  PRIVATE
  ${JNI_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/jvm-packages/xgboost4j/src/native
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/dmlc-core/include
  ${PROJECT_SOURCE_DIR}/rabit/include)

if (ARROW_FOUND AND ARROW_PYTHON_FOUND)
  target_compile_definitions(xgboost4j PRIVATE
                    -D_GLIBCXX_USE_CXX11_ABI=0)
endif (ARROW_FOUND AND ARROW_PYTHON_FOUND)

set_output_directory(xgboost4j ${PROJECT_SOURCE_DIR}/lib)
set_target_properties(
  xgboost4j PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON)
target_link_libraries(xgboost4j PRIVATE ${JAVA_JVM_LIBRARY})
