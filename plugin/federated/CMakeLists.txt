# gRPC needs to be installed first. See README.md.
find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)
find_package(Threads)

# Generated code from the protobuf definition.
add_library(federated_proto federated.proto)
target_link_libraries(federated_proto PUBLIC protobuf::libprotobuf gRPC::grpc gRPC::grpc++)
target_include_directories(federated_proto PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
xgboost_target_properties(federated_proto)

get_target_property(grpc_cpp_plugin_location gRPC::grpc_cpp_plugin LOCATION)
protobuf_generate(TARGET federated_proto LANGUAGE cpp)
protobuf_generate(
    TARGET federated_proto
    LANGUAGE grpc
    GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc
    PLUGIN "protoc-gen-grpc=${grpc_cpp_plugin_location}")

# Wrapper for the gRPC client.
add_library(federated_client INTERFACE)
target_sources(federated_client INTERFACE federated_client.h)
target_link_libraries(federated_client INTERFACE federated_proto)

# Rabit engine for Federated Learning.
target_sources(objxgboost PRIVATE federated_server.cc)
target_link_libraries(objxgboost PRIVATE federated_client)
target_compile_definitions(objxgboost PUBLIC -DXGBOOST_USE_FEDERATED=1)
