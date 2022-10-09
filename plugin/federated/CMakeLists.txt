# Download and build gRPC
if (USE_GRPC_FROM_SYSTEM)
  message(STATUS "Attempting to locate gRPC installation from the system...")
  set(protobuf_MODULE_COMPATIBLE TRUE)
  find_package(Protobuf CONFIG REQUIRED)
  find_package(gRPC CONFIG REQUIRED)
  message(STATUS "Found gRPC: ${gRPC_CONFIG}")
  set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)
  set(_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)
  set(_GRPC_GRPCPP gRPC::grpc++)
  set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:gRPC::grpc_cpp_plugin>)
else ()
  message(STATUS "Downloading gRPC source from GitHub...")
  include(FetchContent)
  FetchContent_Declare(
      grpc
      GIT_REPOSITORY  https://github.com/grpc/grpc.git
      GIT_TAG         v1.49.1
  )
  set(FETCHCONTENT_QUIET OFF)
  FetchContent_MakeAvailable(grpc)
  set(_PROTOBUF_LIBPROTOBUF libprotobuf)
  set(_PROTOBUF_PROTOC $<TARGET_FILE:protoc>)
  set(_GRPC_GRPCPP grpc++)
  set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:grpc_cpp_plugin>)
endif ()

# Proto file
get_filename_component(federated_proto "federated.proto" ABSOLUTE)
get_filename_component(federated_proto_path "${federated_proto}" PATH)

# Generated code from the protobuf definition.
set(federated_srcs "${CMAKE_CURRENT_BINARY_DIR}/federated.pb.cc")
set(federated_hdrs "${CMAKE_CURRENT_BINARY_DIR}/federated.pb.h")
set(federated_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/federated.grpc.pb.cc")
set(federated_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/federated.grpc.pb.h")
add_custom_command(
      OUTPUT "${federated_srcs}" "${federated_hdrs}"
             "${federated_grpc_srcs}" "${federated_grpc_hdrs}"
      COMMAND ${_PROTOBUF_PROTOC}
      ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
           --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
           -I "${federated_proto_path}"
           --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
           "${federated_proto}"
      DEPENDS "${federated_proto}")

add_library(federated_proto_lib STATIC)
target_sources(federated_proto_lib PRIVATE
  ${federated_srcs} ${federated_hdrs}
  ${federated_grpc_srcs} ${federated_grpc_hdrs})
target_link_libraries(federated_proto_lib PUBLIC
  ${_PROTOBUF_LIBPROTOBUF} ${_GRPC_GRPCPP})
target_include_directories(federated_proto_lib PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR})
set_property(TARGET federated_proto_lib PROPERTY POSITION_INDEPENDENT_CODE ON)

# Wrapper for the gRPC client.
add_library(federated_client INTERFACE)
target_sources(federated_client INTERFACE federated_client.h)
target_link_libraries(federated_client INTERFACE federated_proto_lib)

# Rabit engine for Federated Learning.
target_sources(objxgboost PRIVATE federated_server.cc)
target_link_libraries(objxgboost PRIVATE federated_client)
target_compile_definitions(objxgboost PUBLIC -DXGBOOST_USE_FEDERATED=1)
