#define TLAPACK_PREFERRED_MATRIX_LEGACY
#include <tlapack/plugins/legacyArray.hpp>
#include <tlapack/plugins/stdvector.hpp>
#include <tlapack/plugins/float8_iee_p.hpp>
#ifdef USE_MPFR
 #include <tlapack/plugins/mpreal.hpp>
#endif

// <T>LAPACK
#include <tlapack/blas/trsm.hpp>
#include <tlapack/blas/trmm.hpp>
#include <tlapack/lapack/getrf.hpp>
#include <tlapack/lapack/geqr2.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include "../../eigen/Eigen/Core"
#include <tlapack/lapack/ung2r.hpp>


// C++ headers
#include <iostream>
#include <vector>
#include <fstream>


template<typename T>
float run() {

    



}


int main(int argc, char** argv) {

     typedef ml_dtypes::float8_e4m3fn float8e4m3fn;
    typedef ml_dtypes::float8_e5m2 float8e5m2;
    typedef ml_dtypes::float8_ieee_p<4> floate4m3;
    typedef ml_dtypes::float8_ieee_p<3> floate5m2;
    typedef ml_dtypes::block_float8_ieee<4> bfp;
    int n;
    const tlapack::Layout L = tlapack::Layout::ColMajor;
 

    n = atoi(argv[1]);
   
      // Init random seed
    srand(100);
    std::cout.precision(4);
    std::cout << std::scientific << std::showpos;
    

    // printf("run< float, L >( %d )\n", n);
    // run<float, L>(n, 1.0);
    // printf("-----------------------\n");


    //for (int i = 0; i < 1000; i += 10){
   
    //printf("run< float8e4m3fn, L >( %d )\n", n);
    //run<float8e4m3fn , L>(n, ml_dtypes::float8_internal::numeric_limits_float8_e4m3fn::max(), static_cast<float>(i));
    
    // printf("-----------------------\n");

    //  printf("run< float8e5m2, L >( %d )\n", n);
    // if(atoi(argv[5]) == 0)
    // run<floate4m3, L>(n, ml_dtypes::float8_internal::numeric_limits_float8_ieee_p<4>::max()/floate4m3{2.0}, static_cast<float>(atoi(argv[3])));    
    // else if(atoi(argv[5]) == 1)
    run<floate5m2, L>(n, ml_dtypes::float8_internal::numeric_limits_float8_ieee_p<3>::max(), static_cast<float>(atoi(argv[3])));  
    // else if(atoi(argv[5]) == 2)
    // run<float, L>(n,1.0, static_cast<float>(atoi(argv[3])));
    // else if(atoi(argv[5]) == 3)
    // run<bfp, L>(n,bfp(1000.0), static_cast<float>(atoi(argv[3]));
    // else if(atoi(argv[5]) == 4)
    // run<float8e4m3fn, L>(n, ml_dtypes::float8_internal::numeric_limits_float8_e4m3fn::max(), static_cast<float>(atoi(argv[3])));    
    // else 
    // run<int, L>(n,1.0, static_cast<int>(atoi(argv[3])));
    
    
    
   

// #ifdef USE_MPFR
//     printf("run< mpfr::mpreal, L >( %d )\n", n);
//     run<mpfr::mpreal, L>(n, 1);
//     printf("-----------------------\n");

//     printf("run< complex<mpfr::mpreal>, L >( %d )\n", n);
//     run<std::complex<mpfr::mpreal>, L>(n, mpfr::mpreal(1.0));
//     printf("-----------------------\n");
// #endif
    

    return 0;



}