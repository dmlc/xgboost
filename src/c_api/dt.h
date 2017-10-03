
//==============================================================================

/**
 * NA constants
 *
 * Integer-based NAs can be compared by value (e.g. `x == NA_I4`), whereas
 * floating-point NAs require special functions `ISNA_F4(x)` and `ISNA_F8(x)`.
 */

#define BIN_NAF4 0x7F8007A2u
#define BIN_NAF8 0x7FF00000000007A2ull
typedef union { uint64_t i; double d; } double_repr;
typedef union { uint32_t i; float f; } float_repr;
static inline float _nanf_(void) { float_repr x = { BIN_NAF4 }; return x.f; }
static inline double _nand_(void) { double_repr x = { BIN_NAF8 }; return x.d; }
#define NA_I1  (-128)
#define NA_I2  (-32768)
#define NA_I4  (-2147483647-1)
#define NA_I8  (-9223372036854775807-1)
#define NA_U1  255u
#define NA_U2  65535u
#define NA_U4  4294967295u
#define NA_U8  18446744073709551615u
#define NA_F4  _nanf_()
#define NA_F8  _nand_()


#define NA_F4_BITS 0x7F8007A2u
#define NA_F8_BITS 0x7FF00000000007A2ull
extern const int8_t   NA_I1;
extern const int16_t  NA_I2;
extern const int32_t  NA_I4;
extern const int64_t  NA_I8;
extern const uint8_t  NA_U1;
extern const uint16_t NA_U2;
extern const uint32_t NA_U4;
extern const uint64_t NA_U8;
extern       float    NA_F4;
extern       double   NA_F8;

int ISNA_F4(float x);
int ISNA_F8(double x);

#define ISNA_I1(x)  ((int8_t)(x)   == NA_I1)
#define ISNA_I2(x)  ((int16_t)(x)  == NA_I2)
#define ISNA_I4(x)  ((int32_t)(x)  == NA_I4)
#define ISNA_I8(x)  ((int64_t)(x)  == NA_I8)
#define ISNA_U1(x)  ((uint8_t)(x)  == NA_U1)
#define ISNA_U2(x)  ((uint16_t)(x) == NA_U2)
#define ISNA_U4(x)  ((uint32_t)(x) == NA_U4)

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
template<> inline uint8_t  GETNA() { return NA_U1; }
template<> inline uint16_t GETNA() { return NA_U2; }
template<> inline uint32_t GETNA() { return NA_U4; }
template<> inline float    GETNA() { return NA_F4; }
template<> inline double   GETNA() { return NA_F8; }

/**
 * ISNA function
 * Template function that uses the appropriate ISNA_XX macro/function based
 * on the argument type. Returns true if type is invalid.
 */
template <typename T>
inline bool ISNA(T)          { return true;       }
template<> inline bool ISNA(int8_t x)   { return ISNA_I1(x); }
template<> inline bool ISNA(int16_t x)  { return ISNA_I2(x); }
template<> inline bool ISNA(int32_t x)  { return ISNA_I4(x); }
template<> inline bool ISNA(int64_t x)  { return ISNA_I8(x); }
template<> inline bool ISNA(uint8_t x)  { return ISNA_U1(x); }
template<> inline bool ISNA(uint16_t x) { return ISNA_U2(x); }
template<> inline bool ISNA(uint32_t x) { return ISNA_U4(x); }
template<> inline bool ISNA(float x)    { return ISNA_F4(x); }
template<> inline bool ISNA(double x)   { return ISNA_F8(x); }

//==============================================================================
