get_filename_component(CMAKE_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
if(NOT MSVC)
    set(LINT_COMMAND ${CMAKE_SOURCE_DIR}/scripts/lint3.py)
else()
    if((NOT PYTHON2_EXECUTABLE) AND (NOT PYTHON3_EXECUTABLE))
         message(FATAL_ERROR "Cannot lint without python")
    endif()
    # format output so VS can bring us to the offending file/line
	if(PYTHON2_EXECUTABLE)
		set(LINT_COMMAND ${PYTHON2_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/lint.py)
	endif()
    if(PYTHON3_EXECUTABLE)
		set(LINT_COMMAND ${PYTHON3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/lint3.py)
	endif()
endif()

set(LINT_DIRS include src scripts)

cmake_policy(SET CMP0009 NEW)  # suppress cmake warning
execute_process(
    COMMAND ${LINT_COMMAND} ${PROJECT_NAME} all ${LINT_DIRS}
	WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    ERROR_VARIABLE LINT_OUTPUT
    ERROR_STRIP_TRAILING_WHITESPACE
	
)
message(STATUS ${LINT_OUTPUT})