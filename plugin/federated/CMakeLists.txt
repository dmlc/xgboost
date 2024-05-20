# gRPC needs to be installed first. See README.md.
set(protobuf_MODULE_COMPATIBLE TRUE)
set(protobuf_BUILD_SHARED_LIBS TRUE)

find_package(Protobuf CONFIG)
if(NOT Protobuf_FOUND)
  find_package(Protobuf)
endif()
if(NOT Protobuf_FOUND)
  # let CMake emit error
  find_package(Protobuf CONFIG REQUIRED)
endif()

find_package(gRPC CONFIG REQUIRED)
message(STATUS "Found gRPC: ${gRPC_CONFIG}")

# Generated code from the protobuf definition.
add_library(federated_proto STATIC federated.proto)
target_link_libraries(federated_proto PUBLIC protobuf::libprotobuf gRPC::grpc gRPC::grpc++)
target_include_directories(federated_proto PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
xgboost_target_properties(federated_proto)

protobuf_generate(
    TARGET federated_proto
    LANGUAGE cpp
    PROTOC_OUT_DIR "${PROTO_BINARY_DIR}")
protobuf_generate(
    TARGET federated_proto
    LANGUAGE grpc
    GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc
    PLUGIN "protoc-gen-grpc=\$<TARGET_FILE:gRPC::grpc_cpp_plugin>"
    PROTOC_OUT_DIR "${PROTO_BINARY_DIR}")

# Wrapper for the gRPC client.
add_library(federated_client INTERFACE)
target_link_libraries(federated_client INTERFACE federated_proto)

# Rabit engine for Federated Learning.
target_sources(
  objxgboost PRIVATE federated_tracker.cc federated_comm.cc federated_coll.cc
)
if(USE_CUDA)
  target_sources(objxgboost PRIVATE federated_comm.cu federated_coll.cu)
endif()

target_link_libraries(objxgboost PRIVATE federated_client "-Wl,--exclude-libs,ALL")
target_compile_definitions(objxgboost PUBLIC -DXGBOOST_USE_FEDERATED=1)
