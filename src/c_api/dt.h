
//==============================================================================

/**
 * NA constants
 *
 * Integer-based NAs can be compared by value (e.g. `x == NA_I4`), whereas
 * floating-point NAs require special functions `ISNA_F4(x)` and `ISNA_F8(x)`.
 */

//#define BIN_NAF4 0x7F8007A2u
//#define BIN_NAF8 0x7FF00000000007A2ull
//typedef union { uint64_t i; double d; } double_repr;
//typedef union { uint32_t i; float f; } float_repr;
//static inline float _nanf_(void) { float_repr x = { BIN_NAF4 }; return x.f; }
//static inline double _nand_(void) { double_repr x = { BIN_NAF8 }; return x.d; }
#define NA_I1  (-128)
#define NA_I2  (-32768)
#define NA_I4  (-2147483647-1)
#define NA_I8  (-9223372036854775807-1)
//#define NA_U1  255u
//#define NA_U2  65535u
//#define NA_U4  4294967295u
//#define NA_U8  18446744073709551615u
//#define NA_F4  _nanf_()
//#define NA_F8  _nand_()

/**
 * GETNA function
 * Template function that returns the appropriate NA_XX value based on the
 * type of `T`. Returns NULL if `T` is incompatible.
 */
template <typename T>
inline T        GETNA() { return NULL;  }
template<> inline int8_t   GETNA() { return NA_I1; }
template<> inline int16_t  GETNA() { return NA_I2; }
template<> inline int32_t  GETNA() { return NA_I4; }
template<> inline int64_t  GETNA() { return NA_I8; }
//template<> inline uint8_t  GETNA() { return NA_U1; }
//template<> inline uint16_t GETNA() { return NA_U2; }
//template<> inline uint32_t GETNA() { return NA_U4; }
//template<> inline float    GETNA() { return NA_F4; }
//template<> inline double   GETNA() { return NA_F8; }
