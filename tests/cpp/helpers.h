#ifndef XGBOOST_TESTS_CPP_HELPERS_H_
#define XGBOOST_TESTS_CPP_HELPERS_H_

#include <iostream>
#include <fstream>
#include <cstdio>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>

#include <gtest/gtest.h>

std::string TempFileName();

bool FileExists(const std::string name);

long GetFileSize(const std::string filename);

std::string CreateSimpleTestData();

#endif
