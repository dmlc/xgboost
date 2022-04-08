find_package(protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)
find_package(Threads)

add_library(federation_proto OBJECT federation.proto)
target_link_libraries(federation_proto PUBLIC protobuf::libprotobuf gRPC::grpc gRPC::grpc++)
target_include_directories(federation_proto PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

get_target_property(grpc_cpp_plugin_location gRPC::grpc_cpp_plugin LOCATION)
protobuf_generate(TARGET federation_proto LANGUAGE cpp)
protobuf_generate(
    TARGET federation_proto
    LANGUAGE grpc
    GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc
    PLUGIN "protoc-gen-grpc=${grpc_cpp_plugin_location}")

add_executable(federation_server federation_server.cc)
target_link_libraries(federation_server PRIVATE federation_proto)

add_executable(federation_client federation_client.cc)
target_link_libraries(federation_client PRIVATE federation_proto)
