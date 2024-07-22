find_package(JNI REQUIRED)

list(APPEND JVM_SOURCES
  ${PROJECT_SOURCE_DIR}/jvm-packages/xgboost4j/src/native/xgboost4j.cpp
  ${PROJECT_SOURCE_DIR}/jvm-packages/xgboost4j/src/native/xgboost4j-gpu.cpp)

if(USE_CUDA)
  list(APPEND JVM_SOURCES
    ${PROJECT_SOURCE_DIR}/jvm-packages/xgboost4j/src/native/xgboost4j-gpu.cu)
endif()

add_library(xgboost4j SHARED ${JVM_SOURCES} ${XGBOOST_OBJ_SOURCES})

if(ENABLE_ALL_WARNINGS)
  target_compile_options(xgboost4j PUBLIC -Wall -Wextra)
endif()

target_link_libraries(xgboost4j PRIVATE objxgboost)
target_include_directories(xgboost4j
  PRIVATE
  ${JNI_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/jvm-packages/xgboost4j/src/native
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/dmlc-core/include)

set_output_directory(xgboost4j ${PROJECT_SOURCE_DIR}/lib)

# MacOS: Patch libxgboost4j.dylib to use @rpath/libomp.dylib
if(USE_OPENMP AND APPLE)
  patch_openmp_path_macos(xgboost4j libxgboost4j)
endif()
