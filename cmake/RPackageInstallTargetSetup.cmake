# Assembles the R-package files in build_dir;
# if necessary, installs the main R package dependencies;
# runs R CMD INSTALL.
function(setup_rpackage_install_target rlib_target build_dir)
  configure_file(${PROJECT_SOURCE_DIR}/cmake/RPackageInstall.cmake.in ${PROJECT_BINARY_DIR}/RPackageInstall.cmake @ONLY)
  install(
    DIRECTORY "${xgboost_SOURCE_DIR}/R-package"
    DESTINATION "${build_dir}"
    REGEX "src/*" EXCLUDE
    REGEX "R-package/configure" EXCLUDE
  )
  install(TARGETS ${rlib_target}
    LIBRARY DESTINATION "${build_dir}/R-package/src/"
    RUNTIME DESTINATION "${build_dir}/R-package/src/")
  install(SCRIPT ${PROJECT_BINARY_DIR}/RPackageInstall.cmake)
endfunction()