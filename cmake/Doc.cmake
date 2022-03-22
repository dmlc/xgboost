function (run_doxygen)
  find_package(Doxygen REQUIRED)

  if (NOT DOXYGEN_DOT_FOUND)
    message(FATAL_ERROR "Command `dot` not found.  Please install graphviz.")
  endif (NOT DOXYGEN_DOT_FOUND)

  configure_file(
    ${xgboost_SOURCE_DIR}/doc/Doxyfile.in
    ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
  add_custom_target( doc_doxygen ALL
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generate C APIs documentation."
    VERBATIM)
endfunction (run_doxygen)
