
//==============================================================================
#include <wchar.h>

/**
 * NA constants
 *
 * Integer-based NAs can be compared by value (e.g. `x == NA_I4`), whereas
 * floating-point NAs require special functions `ISNA_F4(x)` and `ISNA_F8(x)`.
 */

#define NA_I1  (-128)
#define NA_I2  (-32768)
#define NA_I4  (-2147483647-1)
#define NA_I8  (-9223372036854775807-1)

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



class datacol_struct{
  public:
  double * datacol_double;
  float * datacol_float;
  bool * datacol_bool;
  int8_t * datacol_int1;
  int16_t * datacol_int2;
  int32_t * datacol_int4;
  int64_t * datacol_int8;
  wchar_t * stype;
  int datacoltype;
  int whichj;
  datacol_struct(void **data, const wchar_t ** feature_stypes, int j):
            datacol_bool(reinterpret_cast<bool**>(data)[j]),
            datacol_int1(reinterpret_cast<int8_t**>(data)[j]),
            datacol_int2(reinterpret_cast<int16_t**>(data)[j]),
            datacol_int4(reinterpret_cast<int32_t**>(data)[j]),
            datacol_int8(reinterpret_cast<int64_t**>(data)[j]),
            datacol_double(reinterpret_cast<double**>(data)[j]),
            datacol_float(reinterpret_cast<float**>(data)[j]){

                whichj = j;

                stype = const_cast<wchar_t *>(feature_stypes[j]);

                if(wcscmp(stype,L"f4r")==0){
                   datacoltype = 0;
                }
                else if(wcscmp(stype,L"f8r")==0){
                   datacoltype = 1;
                }
                else if(wcscmp(stype,L"i1b")==0){
                   datacoltype = 2;
                }
                else if(wcscmp(stype,L"i4i")==0){
                   datacoltype = 3;
                }
                else if(wcscmp(stype,L"i1i")==0){
                   datacoltype = 4;
                }
                else if(wcscmp(stype,L"i2i")==0){
                   datacoltype = 5;
                }
                else if(wcscmp(stype,L"i8i")==0){
                   datacoltype = 6;
                }
                else{
                    fwprintf(stderr,L"Unknown type %s", stype);
                    exit(1);
                }
            };
};


bool dt_is_missing_and_get_value(datacol_struct *d, int i, float *value)
{
    // order of likelihood
    switch(d->datacoltype){
        case 0:
            if(!std::isfinite(d->datacol_float[i])) return true;
            *value = static_cast<float>(d->datacol_float[i]);
            return false;
            break;
        case 1:
            if(!std::isfinite(d->datacol_double[i])) return true;
            *value = static_cast<float>(d->datacol_double[i]);
            return false;
            break;
        case 2:
            if(d->datacol_bool[i]==GETNA<bool>()) return true;
            *value = static_cast<float>(d->datacol_bool[i]);
            return false;
            break;
        case 3:
            if(d->datacol_int4[i]==GETNA<int32_t>()) return true;
            *value = static_cast<float>(d->datacol_int4[i]);
            return false;
            break;
        case 4:
            if(d->datacol_int1[i]==GETNA<int8_t>()) return true;
            *value = static_cast<float>(d->datacol_int1[i]);
            return false;
            break;
        case 5:
            if(d->datacol_int2[i]==GETNA<int16_t>()) return true;
            *value = static_cast<float>(d->datacol_int2[i]);
            return false;
            break;
        case 6:
            if(d->datacol_int8[i]==GETNA<int64_t>()) return true;
            *value = static_cast<float>(d->datacol_int8[i]);
            return false;
            break;
        default:
            wprintf(L"Unknown type %s", d->stype);
            fprintf(stderr,"Unknown datacoltype=%d\n",d->datacoltype);
            fflush(stderr);
            fflush(stdout);
            exit(1);
    }
}
