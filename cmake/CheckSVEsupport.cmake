function(check_xgboost_sve_support)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    include(CheckCSourceCompiles)
    
    # Save the original C_FLAGS to restore later
    set(ORIGINAL_C_FLAGS "${CMAKE_C_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8-a+sve")

    # Check if the compiler supports ARM SVE
    check_c_source_compiles("
    #if defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
    #include <arm_sve.h>
    int main() {
        svfloat64_t a;
        a = svdup_n_f64(0);
        return 0;
    }
    #endif
    " XGBOOST_COMPILER_HAS_ARM_SVE)

    if(XGBOOST_COMPILER_HAS_ARM_SVE)
        message(STATUS "ARM SVE compiler support detected")

        # Check for hardware support
        set(SOURCE_CODE "
        #include <sys/prctl.h>
        int main() {
            int ret = prctl(PR_SVE_GET_VL);
            return ret >= 0 ? 0 : 1;
        }
        ")
        file(WRITE ${CMAKE_BINARY_DIR}/check_sve_support.c "${SOURCE_CODE}")
        try_run(RUN_RESULT COMPILE_RESULT
            ${CMAKE_BINARY_DIR}/check_sve_support_output
            ${CMAKE_BINARY_DIR}/check_sve_support.c
        )

        if(RUN_RESULT EQUAL 0)
            message(STATUS "ARM SVE hardware support detected")
            # Apply the SVE flags and definitions specifically to the xgboost target
            set(XGBOOST_ARM_SVE_HARDWARE_SUPPORT TRUE PARENT_SCOPE)
            set(XGBOOST_SVE_FLAGS "-march=armv8-a+sve" PARENT_SCOPE)
            set(XGBOOST_SVE_DEFINITIONS "-DXGBOOST_SVE_SUPPORT_DETECTED" PARENT_SCOPE)
        else()
            message(STATUS "ARM SVE hardware support not detected")
        endif()
    else()
        message(STATUS "ARM SVE compiler support not detected")
    endif()

    # Restore the original C_FLAGS
    set(CMAKE_C_FLAGS "${ORIGINAL_C_FLAGS}")
else()
    message(STATUS "Not an aarch64 architecture")
endif()
endfunction()