function (find_prefetch_intrinsics)
  include(CheckCXXSourceCompiles)
  check_cxx_source_compiles("
  #include <xmmintrin.h>
  int main() {
    char data = 0;
    const char* address = &data;
    _mm_prefetch(address, _MM_HINT_NTA);
    return 0;
  }
  " XGBOOST_MM_PREFETCH_PRESENT)
  check_cxx_source_compiles("
  int main() {
    char data = 0;
    const char* address = &data;
    __builtin_prefetch(address, 0, 0);
    return 0;
  }
  " XGBOOST_BUILTIN_PREFETCH_PRESENT)
  set(XGBOOST_MM_PREFETCH_PRESENT ${XGBOOST_MM_PREFETCH_PRESENT} PARENT_SCOPE)
  set(XGBOOST_BUILTIN_PREFETCH_PRESENT ${XGBOOST_BUILTIN_PREFETCH_PRESENT} PARENT_SCOPE)
endfunction (find_prefetch_intrinsics)
