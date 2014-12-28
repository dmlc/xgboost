/*!
 *  Copyright (c) 2014 by Contributors
 * \file engine_mock.cc
 * \brief this is an engine implementation that will 
 * insert failures in certain call point, to test if the engine is robust to failure
 * \author Tianqi Chen
 */
// define use MOCK, os we will use mock Manager
#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
#define NOMINMAX
// switch engine to AllreduceMock
#define RABIT_USE_MOCK
#include "./allreduce_mock.h"
#include "./engine.cc"

