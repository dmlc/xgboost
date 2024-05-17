/**
 * Copyright 2014-2024 by XGBoost Contributors
 */

#include <iostream>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "./processor.h"
#include "plugins/mock_processor.h"

namespace processing {
    using LoadFunc = Processor *(const char *);

    Processor* ProcessorLoader::load(const std::string& plugin_name) {
        // Dummy processor for unit testing without loading a shared library
        if (plugin_name == kMockProcessor) {
            return new MockProcessor();
        }

        auto lib_name = "libproc_" + plugin_name;

        auto extension =
#if defined(_WIN32) || defined(_WIN64)
            ".dll";
#elif defined(__APPLE__) || defined(__MACH__)
            ".dylib";
#else
            ".so";
#endif
        auto lib_file_name = lib_name + extension;

        std::string lib_path;

        if (params.find(kLibraryPath) == params.end()) {
            lib_path = lib_file_name;
        } else {
            auto p = params[kLibraryPath];
            if (p.back() != '/' && p.back() != '\\') {
                p += '/';
            }
            lib_path = p + lib_file_name;
        }

#if defined(_WIN32) || defined(_WIN64)
        handle_ = reinterpret_cast<void *>(LoadLibrary(lib_path.c_str()));
        if (!handle_) {
            std::cerr << "Failed to load the dynamic library" << std::endl;
            return NULL;
        }

        void* func_ptr = reinterpret_cast<void *>(GetProcAddress((HMODULE)handle_, kLoadFunc));
        if (!func_ptr) {
            std::cerr << "Failed to find loader function." << std::endl;
            return NULL;
        }
#else
        handle_ = dlopen(lib_path.c_str(), RTLD_LAZY);
        if (!handle_) {
            std::cerr << "Failed to load the dynamic library: " << dlerror() << std::endl;
            return NULL;
        }
        void* func_ptr = dlsym(handle_, kLoadFunc);
        if (!func_ptr) {
            std::cerr << "Failed to find loader function: " << dlerror() << std::endl;
            return NULL;
        }
#endif

        auto func = reinterpret_cast<LoadFunc *>(func_ptr);

        return (*func)(plugin_name.c_str());
    }

    void ProcessorLoader::unload() {
#if defined(_WIN32)
        if (handle_) {
            FreeLibrary(handle_);
        }
#else
        if (handle_) {
            dlclose(handle_);
        }
#endif
    }
}  // namespace processing
