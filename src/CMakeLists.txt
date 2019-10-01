option(DMLC_ROOT "Specify root of external dmlc core.")

add_library(allreduce_base "")
add_library(allreduce_mock "")

target_sources(
        allreduce_base
        PRIVATE
        allreduce_base.cc
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/allreduce_base.h
)
target_sources(
	allreduce_mock
	PRIVATE
	allreduce_robust.cc
	PUBLIC
	${CMAKE_CURRENT_LIST_DIR}/allreduce_mock.h
)

target_include_directories(
        allreduce_base
        PUBLIC
	${DMLC_ROOT}/include
        ${CMAKE_CURRENT_LIST_DIR}/../../include)

target_include_directories(
	allreduce_mock
	PUBLIC
	${DMLC_ROOT}/include
	${CMAKE_CURRENT_LIST_DIR}/../../include)
