find_package(protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)
find_package(Threads)

add_library(federated_proto federated.proto)
target_link_libraries(federated_proto PUBLIC protobuf::libprotobuf gRPC::grpc gRPC::grpc++)
target_include_directories(federated_proto PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
set_property(TARGET federated_proto PROPERTY POSITION_INDEPENDENT_CODE ON)

get_target_property(grpc_cpp_plugin_location gRPC::grpc_cpp_plugin LOCATION)
protobuf_generate(TARGET federated_proto LANGUAGE cpp)
protobuf_generate(
    TARGET federated_proto
    LANGUAGE grpc
    GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc
    PLUGIN "protoc-gen-grpc=${grpc_cpp_plugin_location}")

add_library(federated_client INTERFACE federated_client.h)
target_link_libraries(federated_client INTERFACE federated_proto)

add_executable(federated_server federated_server.cc)
target_link_libraries(federated_server PRIVATE federated_proto)

add_executable(test_client test_client.cc)
target_link_libraries(test_client PRIVATE federated_client)

target_sources(objxgboost PRIVATE engine_federated.cc)
target_link_libraries(objxgboost PRIVATE federated_client)
target_compile_definitions(objxgboost PUBLIC -DXGBOOST_USE_FEDERATED=1)
