// #ifndef TLAPACK_FLOAT8_HH
// #define TLAPACK_FLOAT8_HH
#include <math.h>

#include <complex>
#include <limits>
#include <ostream>
#include <type_traits>

#include "../../../eigen/Eigen/Core"
#include "float8.h"
#include "tlapack/base/scalar_type_traits.hpp"
#include "tlapack/base/types.hpp"

using namespace std;
using namespace ml_dtypes::float8_internal;






namespace tlapack {
    namespace traits {
        template <>
        struct real_type_traits<ml_dtypes::float8_internal::float8_e4m3fn, int> {
            using type = ml_dtypes::float8_internal::float8_e4m3fn;
            constexpr static bool is_real = true;
        };
        template <>
        struct complex_type_traits<ml_dtypes::float8_internal::float8_e4m3fn, int> {
            using type = std::complex<ml_dtypes::float8_internal::float8_e4m3fn>;
            constexpr static bool is_complex = false;
        };
    }  // namespace traits


  }

namespace tlapack {
    namespace traits {
        template <int p>
        struct real_type_traits<ml_dtypes::float8_internal::float8_ieee_p<p>, int> {
            using type = ml_dtypes::float8_internal::float8_ieee_p<p>;
            constexpr static bool is_real = true;
        };
        template <int p>
        struct complex_type_traits<ml_dtypes::float8_internal::float8_ieee_p<p>, int> {
            using type = std::complex<ml_dtypes::float8_internal::float8_ieee_p<p>>;
            constexpr static bool is_complex = false;
        };
    }  // namespace traits

  }

namespace tlapack {
    namespace traits {
        template <>
        struct real_type_traits<ml_dtypes::float8_internal::float8_e5m2, int> {
            using type = ml_dtypes::float8_internal::float8_e5m2;
            constexpr static bool is_real = true;
        };
        template <>
        struct complex_type_traits<ml_dtypes::float8_internal::float8_e5m2, int> {
            using type = std::complex<ml_dtypes::float8_internal::float8_e5m2>;
            constexpr static bool is_complex = false;
        };
    }  // namespace traits

    
  }

namespace tlapack {
    namespace traits {
        template <>
        struct real_type_traits<ml_dtypes::float8_internal::float8_e3m4, int> {
            using type = ml_dtypes::float8_internal::float8_e3m4;
            constexpr static bool is_real = true;
        };
        template <>
        struct complex_type_traits<ml_dtypes::float8_internal::float8_e3m4, int> {
            using type = std::complex<ml_dtypes::float8_internal::float8_e3m4>;
            constexpr static bool is_complex = false;
        };
    }  // namespace traits

    
  }

namespace tlapack {
    namespace traits {
        template <int p>
        struct real_type_traits<ml_dtypes::block_float8_ieee<p>, int> {
            using type = ml_dtypes::block_float8_ieee<p>;
            constexpr static bool is_real = true;
        };
        template <int p>
        struct complex_type_traits<ml_dtypes::block_float8_ieee<p>, int> {
            using type = std::complex<ml_dtypes::block_float8_ieee<p>>;
            constexpr static bool is_complex = false;
        };
    }  // namespace traits

    template <int p>
    int get_scaling_unit(ml_dtypes::block_float8_ieee<p> x) {
        return x.scaling_unit;
    }
    template <int p>
    float get_float_part(ml_dtypes::block_float8_ieee<p> x) {
        return x.float_part;
    }
    template <int p>
    void set_scaling_unit(ml_dtypes::block_float8_ieee<p> x, int s) {
        x.scaling_unit = s;
        return;
    }
    template <int p>
    void set_float_part(ml_dtypes::block_float8_ieee<p> x, ml_dtypes::float8_ieee_p<p> fl) {
        x.float_part = fl;
        return;
    }

  }

  // namespace tlapack

inline std::istream& operator>>(std::istream& is, ml_dtypes::float8_e4m3fn& x)
{
    float f;
    is >> f;
    x = ml_dtypes::float8_e4m3fn(f);
    return is;
}

inline std::istream& operator>>(std::istream& is, ml_dtypes::float8_e5m2& x)
{
     float f;
    is >> f;
    x = ml_dtypes::float8_e5m2(f);
    return is;
}

inline std::istream& operator>>(std::istream& is, ml_dtypes::float8_e3m4& x)
{
     float f;
    is >> f;
    x = ml_dtypes::float8_e3m4(f);
    return is;
}

template<int p>
inline std::istream& operator>>(std::istream& is, ml_dtypes::float8_ieee_p<p>& x)
{
     float f;
    is >> f;
    x = ml_dtypes::float8_ieee_p<p>(f);
    return is;
}


template<int p>
inline std::istream& operator>>(std::istream& is, ml_dtypes::float8_internal::block_float8_ieee<p>& x)
{
     float f;
    is >> f;
    x = ml_dtypes::float8_internal::block_float8_ieee<p>(f);
    return is;
}



using namespace tlapack;
  namespace ml_dtypes{
    namespace float8_internal {
        typedef float8_e4m3fn float8e4m3fn;
        inline float8e4m3fn ceil(float8e4m3fn x) noexcept
    {
        return float8e4m3fn(ConstexprCeil(double(x)));
    }
    inline float8e4m3fn floor(float8e4m3fn x) noexcept
    {
        return float8e4m3fn(-ConstexprCeil(-1 * double(x)));
    }
    inline float8e4m3fn log2(float8e4m3fn x) noexcept
    {
        return float8e4m3fn(log(double(x)));
    }
    inline float8e4m3fn max(float8e4m3fn x, float8e4m3fn y) noexcept
    {
        return x > y ? x : y;
    }
    inline float8e4m3fn min(float8e4m3fn x, float8e4m3fn y) noexcept
    {
        return x > y ? y : x;
    }
    inline float8e4m3fn sqrt(float8e4m3fn x) noexcept
    { 
        return float8e4m3fn(std::sqrt(double(x)));
    }
    inline float8e4m3fn pow(int x, float8e4m3fn y)
    {
        return float8e4m3fn(std::pow(double(x), double(y)));
    }

    // e5m2
    typedef float8_e5m2 float8e5m2;
    inline float8e5m2 ceil(float8e5m2 x) noexcept
    {
        return float8e5m2(ConstexprCeil(double(x)));
    }
    inline float8e5m2 floor(float8e5m2 x) noexcept
    {
        return float8e5m2(-ConstexprCeil(-1 * double(x)));
    }
    inline float8e5m2 log2(float8e5m2 x) noexcept
    {
        return float8e5m2(log(double(x)));
    }
    inline float8e5m2 max(float8e5m2 x, float8e5m2 y) noexcept
    {
        return x > y ? x : y;
    }
    inline float8e5m2 min(float8e5m2 x, float8e5m2 y) noexcept
    {
        return x > y ? y : x;
    }
    inline float8e5m2 sqrt(float8e5m2 x) noexcept
    {
        return float8e5m2(std::sqrt(double(x)));
    }
    inline float8e5m2 pow(int x, float8e5m2 y)
    {
        return float8e5m2(std::pow(double(x), double(y)));
    }

    // e3m4
    typedef float8_e3m4 float8e3m4;
    inline float8e3m4 ceil(float8e3m4 x) noexcept
    {
        return float8e3m4(ConstexprCeil(double(x)));
    }
    inline float8e3m4 floor(float8e3m4 x) noexcept
    {
        return float8e3m4(-ConstexprCeil(-1 * double(x)));
    }
    inline float8e3m4 log2(float8e3m4 x) noexcept
    {
        return float8e3m4(log(double(x)));
    }
    inline float8e3m4 max(float8e3m4 x, float8e3m4 y) noexcept
    {
        return x > y ? x : y;
    }
    inline float8e3m4 min(float8e3m4 x, float8e3m4 y) noexcept
    {
        return x > y ? y : x;
    }
    inline float8e3m4 sqrt(float8e3m4 x) noexcept
    {
        return float8e3m4(std::sqrt(double(x)));
    }
    inline float8e3m4 pow(int x, float8e3m4 y)
    {
        return float8e3m4(std::pow(double(x), double(y)));
    }


    //ieee p3109
    template<int p>
    inline float8_ieee_p<p> ceil(float8_ieee_p<p> x) noexcept
    {
        return float8_ieee_p<p>(ConstexprCeil(double(x)));
    }

    template<int p>
    inline float8_ieee_p<p> floor(float8_ieee_p<p> x) noexcept
    {
        return float8_ieee_p<p>(-ConstexprCeil(-1 * double(x)));
    }

    template<int p>
    inline float8_ieee_p<p> log2(float8_ieee_p<p> x) noexcept
    {
        return float8_ieee_p<p>(log(double(x)));
    }


    template<int p>
    inline float8_ieee_p<p> max(float8_ieee_p<p> x, float8_ieee_p<p> y) noexcept
    {
        return x > y ? x : y;
    }

    template<int p>
    inline float8_ieee_p<p> min(float8_ieee_p<p> x, float8_ieee_p<p> y) noexcept
    {
        return x > y ? y : x;
    }


    template<int p>
    inline float8_ieee_p<p> sqrt(float8_ieee_p<p> x) noexcept
    {
        return float8_ieee_p<p>(std::sqrt(double(x)));
    }


    template<int p>
    inline float8_ieee_p<p> pow(int x, float8_ieee_p<p> y)
    {
        return float8_ieee_p<p>(std::pow(double(x), double(y)));
    }



    //bfp p3109
    template<int p>
    inline block_float8_ieee<p> ceil(block_float8_ieee<p> x) noexcept
    {
        return block_float8_ieee<p>(ConstexprCeil(double(x)));
    }

    template<int p>
    inline block_float8_ieee<p> floor(block_float8_ieee<p> x) noexcept
    {
        return block_float8_ieee<p>(-ConstexprCeil(-1 * double(x)));
    }

    template<int p>
    inline block_float8_ieee<p> log2(block_float8_ieee<p> x) noexcept
    {
        return block_float8_ieee<p>(log(double(x)));
    }


    template<int p>
    inline block_float8_ieee<p> max(block_float8_ieee<p> x, block_float8_ieee<p> y) noexcept
    {
        return double(x) > double(y) ? x : y;
     
    }

    template<int p>
    inline block_float8_ieee<p> min(block_float8_ieee<p> x, block_float8_ieee<p> y) noexcept
    {
        return double(x) > double(y) ? y : x;
    }


    template<int p>
    inline block_float8_ieee<p> sqrt(block_float8_ieee<p> x) noexcept
    {
        return block_float8_ieee<p> (std::sqrt(double(x)));
    }


    template<int p>
    inline block_float8_ieee<p> pow(int x, block_float8_ieee<p> y)
    {
        return block_float8_ieee<p> (std::pow(double(x), double(y)));
    }

    template<int p>
    inline block_float8_ieee<p> abs(block_float8_ieee<p> y)
    {
        block_float8_ieee<p> to_ret = y;
        to_ret.float_part = abs(y.float_part);
        return to_ret;
    }

   
    }

    }

template<int p>
bool is_bfp(ml_dtypes::block_float8_ieee<p>){
    return true;
}




