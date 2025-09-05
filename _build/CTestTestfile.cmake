# CMake generated Testfile for 
# Source directory: /export/users/drazdobu/xgboost
# Build directory: /export/users/drazdobu/xgboost/_build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(TestXGBoostLib "/export/users/drazdobu/xgboost/_build/testxgboost")
set_tests_properties(TestXGBoostLib PROPERTIES  WORKING_DIRECTORY "/export/users/drazdobu/xgboost/_build" _BACKTRACE_TRIPLES "/export/users/drazdobu/xgboost/CMakeLists.txt;520;add_test;/export/users/drazdobu/xgboost/CMakeLists.txt;0;")
subdirs("dmlc-core")
subdirs("src")
subdirs("plugin")
subdirs("tests/cpp")
