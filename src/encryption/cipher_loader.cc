#include <iostream>

// Platform-specific headers
#if defined(_WIN32)
    #include <windows.h>
#else
    #include <dlfcn.h>
#endif

#include "cipher.h"

namespace xgboost::encryption {

    Cipher Cipher::load(std::string plugin_name) {


        lib_name = "cipher_" + plugin_name;
        const char* extension =
#if defined(_WIN32)
            ".dll";
#elif defined(__APPLE__) || defined(__MACH__)
            ".dylib";
#else
            ".so";
#endif
            // Load the dynamic library based on the platform
#if defined(_WIN32)
    HMODULE handle = LoadLibrary((libraryName + std::to_string(VERSION_NUMBER) + extension).c_str());
    if (!handle) {
            std::cerr << "Failed to load the dynamic library" << std::endl;
            return 1;
    }
#elif defined(__APPLE__) || defined(__MACH__)
    void* handle = dlopen((libraryName + std::to_string(VERSION_NUMBER) + extension).c_str(), RTLD_LAZY);
    if (!handle) {
        std::cerr << "Failed to load the dynamic library: " << dlerror() << std::endl;
        return 1;
    }
#else
            void* handle = dlopen((libraryName + std::to_string(VERSION_NUMBER) + extension).c_str(), RTLD_LAZY);
        if (!handle) {
            std::cerr << "Failed to load the dynamic library: " << dlerror() << std::endl;
            return 1;
        }
#endif

            // Use the loaded library
            // For example, you can use platform-specific code to retrieve function pointers and call them

            // Close the library when done
#if defined(_WIN32)
            FreeLibrary(handle);
#else
            dlclose(handle);
#endif

            return 0;
        }
    }

}