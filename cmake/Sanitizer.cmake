# Set appropriate compiler and linker flags for sanitizers.
#
# Usage of this module:
#  enable_sanitizers("address;leak")

# Add flags
macro(enable_sanitizer santizer)
  if(${santizer} MATCHES "address")
    find_package(ASan REQUIRED)
    set(SAN_COMPILE_FLAGS "${SAN_COMPILE_FLAGS} -fsanitize=address")
    link_libraries(${ASan_LIBRARY})

  elseif(${santizer} MATCHES "thread")
    find_package(TSan REQUIRED)
    set(SAN_COMPILE_FLAGS "${SAN_COMPILE_FLAGS} -fsanitize=thread")
    link_libraries(${TSan_LIBRARY})

  elseif(${santizer} MATCHES "leak")
    find_package(LSan REQUIRED)
    set(SAN_COMPILE_FLAGS "${SAN_COMPILE_FLAGS} -fsanitize=leak")
    link_libraries(${LSan_LIBRARY})

  else()
    message(FATAL_ERROR "Santizer ${santizer} not supported.")
  endif()
endmacro()

macro(enable_sanitizers SANITIZERS)
  # Check sanitizers compatibility.
  # Idealy, we should use if(san IN_LIST SANITIZERS) ... endif()
  # But I haven't figure out how to make it work.
  foreach ( _san ${SANITIZERS} )
    string(TOLOWER ${_san} _san)
    if (_san MATCHES "thread")
      if (${_use_other_sanitizers})
        message(FATAL_ERROR
          "thread sanitizer is not compatible with ${_san} sanitizer.")
      endif()
      set(_use_thread_sanitizer 1)
    else ()
      if (${_use_thread_sanitizer})
        message(FATAL_ERROR
          "${_san} sanitizer is not compatible with thread sanitizer.")
      endif()
      set(_use_other_sanitizers 1)
    endif()
  endforeach()

  message("Sanitizers: ${SANITIZERS}")

  foreach( _san ${SANITIZERS} )
    string(TOLOWER ${_san} _san)
    enable_sanitizer(${_san})
  endforeach()
  message("Sanitizers compile flags: ${SAN_COMPILE_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SAN_COMPILE_FLAGS}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SAN_COMPILE_FLAGS}")
endmacro()
