/**
 * Copyright 2014-2024 by XGBoost Contributors
 */
#include <iostream>
#include <dlfcn.h>

#include "./processor.h"
#include "plugins/dummy_processor.h"

namespace xgboost::processing {
    using LoadFunc = Processor *(const char *);

    Processor* ProcessorLoader::load(const std::string& plugin_name) {
        // Dummy processor for unit testing without loading a shared library
        if (plugin_name == kDummyProcessor) {
            return new DummyProcessor();
        }

        auto lib_name = "libproc_" + plugin_name;

        auto extension =
#if defined(__APPLE__) || defined(__MACH__)
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
            if (p.back() != '/') {
                p += '/';
            }
            lib_path = p + lib_file_name;
        }

        handle = dlopen(lib_path.c_str(), RTLD_LAZY);
        if (!handle) {
            std::cerr << "Failed to load the dynamic library: " << dlerror() << std::endl;
            return NULL;
        }

        void* func_ptr = dlsym(handle, kLoadFunc);

        if (!func_ptr) {
            std::cerr << "Failed to find loader function: " << dlerror() << std::endl;
            return NULL;
        }

        auto func = reinterpret_cast<LoadFunc *>(func_ptr);

        return (*func)(plugin_name.c_str());
    }

    void ProcessorLoader::unload() {
        dlclose(handle);
    }
}  // namespace xgboost::processing
