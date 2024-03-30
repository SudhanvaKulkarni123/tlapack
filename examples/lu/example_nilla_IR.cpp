#define PY_SSIZE_T_CLEAN
#define TLAPACK_PREFERRED_MATRIX_LEGACY
#include <tlapack/plugins/legacyArray.hpp>
#include <tlapack/plugins/stdvector.hpp>
#include <tlapack/plugins/float8_iee_p.hpp>
#include <Python.h>
// <T>LAPACK
#include <tlapack/blas/trsm.hpp>
#include <tlapack/blas/trsv.hpp>
#include <tlapack/blas/trmm.hpp>
#include <tlapack/lapack/getrf.hpp>
#include <tlapack/lapack/geqr2.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include "../../eigen/Eigen/Core"
#include <tlapack/lapack/ung2r.hpp>
#include <tlapack/blas/axpy.hpp>


// C++ headers
#include <iostream>
#include <vector>
#include <fstream>

template <typename matrix_t>
void printMatrix(const matrix_t& A)
{
    using idx_t = tlapack::size_type<matrix_t>;
    const idx_t m = tlapack::nrows(A);
    const idx_t n = tlapack::ncols(A);

    for (idx_t i = 0; i < m; ++i) {
        std::cout << std::endl;
        for (idx_t j = 0; j < n; ++j)
            std::cout << A(i, j) << " ";
    }
}
//------------------------------------------------------------------------------

template<typename matrix_t>
void constructMatrix(matrix_t& A, float cond) {
    //this is an ambitious function that uses a Python embedding to call the functions found in generate\ copy.py to fill in the entries of A
    return;

}

std::ofstream myfile("e5m2_error_f_cond.csv");
template <class T>
double run(size_t n, T scale, float cond) {

    //this function will generate a random b, random A and solve Ax = b in 8 bit first and then use IR in higher precisions.
    using matrix_t = tlapack::LegacyMatrix<T>;
    using real_t = tlapack::real_type<T>;
    using idx_t = size_t;
    using range = std::__1::pair<idx_t, idx_t>;

    // Functors for creating new matrices
    tlapack::Create<matrix_t> new_matrix;

    // Create the n-by-n matrix A
    std::vector<T> A_(n * n);
    tlapack::LegacyMatrix<T, idx_t> A(n, n, A_.data(), n);

    std::vector<float> FG_(n * n);
    tlapack::LegacyMatrix<float, idx_t> FG(n, n, FG_.data(), n);


    
    for(int j = 0; j < n; ++j){
        for(int i = 0; i < n; ++i){
            FG(i,j) = (static_cast<float>(rand()))/static_cast<float>(RAND_MAX);
            A(i,j) = static_cast<T>(FG(i,j));
        }
    }

    //now generate n dimensional vector b to solve Ax = b
    std::vector<float> b_(n);
    tlapack::LegacyVector<float, idx_t> b(n, b_.data());

    std::vector<float> b1_(n);
    tlapack::LegacyVector<float, idx_t> b1(n, b1_.data());

    std::vector<float> b2_(n);
    tlapack::LegacyVector<float, idx_t> b2(n, b2_.data());

    std::vector<float> r_(n);
    tlapack::LegacyVector<float, idx_t> r(n, r_.data());

    for( int i = 0; i < n; i++) {
        b[i] = (static_cast<float>(rand()))/static_cast<float>(RAND_MAX);
        b1[i] = b[i];
        b2[i] = b[i];
    }

    //perform LU on A and FG
    std::vector<T> LU_(n * n);
    tlapack::LegacyMatrix<T, idx_t> LU(n, n, LU_.data(), n);

    std::vector<float> LU_float_(n * n);
    tlapack::LegacyMatrix<float, idx_t> LU_float(n, n, LU_float_.data(), n);

   
    // Matrix A is kept unchanged
    tlapack::lacpy(tlapack::GENERAL, A, LU);
    tlapack::lacpy(tlapack::GENERAL, FG, LU_float);

    //declare arrays for piv
    std::vector<size_t> piv_lo(n);
    std::vector<size_t> piv_hi(n);

    int info = tlapack::getrf(LU, piv_lo);
    if (info != 0) {
        std::cerr << "Matrix could not be factorized :(" << std::endl;
        return -1;
    }
    int infotoo = tlapack::getrf(LU_float, piv_hi);
    if (infotoo != 0) {
        std::cerr << "Matrix could not be factorized in fp32 :(" << std::endl;
        return -1;
    }


    //create copy to stor low prec LU in high prec
    std::vector<float> LU_copy_(n * n);
    tlapack::LegacyMatrix<float, idx_t> LU_copy(n, n, LU_copy_.data(), n);

     for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            LU_copy(i,j) = static_cast<float>(LU(i,j));
        }
    }

    for (idx_t i = 0; i < n;i++){
        if (piv_lo[i] != i) {
            auto tmp = b1[piv_lo[i]];
            b1[piv_lo[i]] = b1[i];
            b1[i] = tmp;
        }
    }

    tlapack::trsv(Uplo::Lower, NO_TRANS, Diag::Unit, LU_copy, b1);
    tlapack::trsv(Uplo::Upper, NO_TRANS, Diag::NonUnit, LU_copy, b1);

    std::cout << "b1 before we begin IR :" << b1[0] << "," << b1[1] << "," << b1[2] << "," << b1[3] << "," << b1[4] << "," << std::endl;

    //now we can begin the actual IR
    float res_norm = 1.0;

    int count = 0; //to keep track of number of IR iterations

    
    
    //b1 is x_0  
    do {
        //initiaize r = b
        count = count + 1;
        for(int i = 0; i < n; i++) 
        {
        r[i] = b[i];
        }
        //r_i = Ax_i - b
        tlapack::gemv(tlapack::NO_TRANS, 1.0, FG, b1, -1.0, r);  
        //get norm of r
        res_norm = tlapack::nrm2(r);   
        //solve Ad_{i+1} = r_i
        for (idx_t i = 0; i < n;i++){
        if (piv_lo[i] != i) {
            auto tmp = r[piv_lo[i]];
            r[piv_lo[i]] = r[i];
            r[i] = tmp;
            }
        }
        tlapack::trsv(Uplo::Lower, NO_TRANS, Diag::Unit, LU_copy, r);
        tlapack::trsv(Uplo::Upper, NO_TRANS, Diag::NonUnit, LU_copy, r);
        //now d_{i+1} is stored in r
        // x_{i+1} = x_i - d_{i+1}
        tlapack::axpy(-1.0, r, b1);
        //x_{i+1} is now stored in b1
        if(count > 10) break;
        
    } while(res_norm > 0.000001);


    for (idx_t i = 0; i < n;i++){
        if (piv_hi[i] != i) {
            auto tmp = b2[piv_hi[i]];
            b2[piv_hi[i]] = b2[i];
            b2[i] = tmp;
        }
    }

    tlapack::trsv(Uplo::Lower, NO_TRANS, Diag::Unit, LU_float, b2);
    tlapack::trsv(Uplo::Upper, NO_TRANS, Diag::NonUnit, LU_float, b2);

    std::cout << "result :" << b1[0] << "," << b1[1] << "," << b1[2] << "," << b1[3] << "," << b1[4] << "," << std::endl;
    std::cout << "other result :" << b2[0] << "," << b2[1] << "," << b2[2] << "," << b2[3] << "," << b2[4] << "," << std::endl;

    auto nrmb2 = tlapack::nrm2(b2);
    //auto nrmb2 = find_max(b2, n);
    tlapack::axpy(-1.0,b1,b2);
    for(int i = 0; i < n; i++) {
        if(isnan(b2[i])) std::cout << "found nan" << std::endl;
    }
    //return find_max(b2, n)/nrmb2;
    //return tlapack::nrm2(b2)/nrmb2;
    //std::cout << "error in inf norm" << tlapack::nrm2(b2)/nrmb2 << std::endl;
    std::cout << "residual err :" << res_norm << std::endl;
    std::cout << "error in 2 norm : " << tlapack::nrm2(b2)/nrmb2 << std::endl;
    return count;



    
}

int main(int argc, char** argv)
{

    typedef ml_dtypes::float8_e4m3fn float8e4m3fn;
    typedef ml_dtypes::float8_e5m2 float8e5m2;
    typedef ml_dtypes::float8_ieee_p<4> floate4m3;
    typedef ml_dtypes::float8_ieee_p<3> floate5m2;
    typedef ml_dtypes::block_float8_ieee<4> bfp;
    int m, n;


    // Default arguments
    n = atoi(argv[1]);
    double err1 = 0;
    double er3 = 0;
    double err2 = 0;

    if(atoi(argv[2]) == 0)
    er3 += run<floate4m3>(n, ml_dtypes::float8_internal::numeric_limits_float8_ieee_p<4>::max()/floate4m3{2.0}, static_cast<float>(atoi(argv[3])));    
    else if(atoi(argv[2]) == 1)
    er3 += run<floate5m2>(n, ml_dtypes::float8_internal::numeric_limits_float8_ieee_p<3>::max(), static_cast<float>(atoi(argv[3])));  
    else if(atoi(argv[2]) == 2)
    er3 +=   run<float>(n,1.0, static_cast<float>(atoi(argv[3])));
    else if(atoi(argv[2]) == 3)
    er3 += run<bfp>(n,bfp(1000.0), static_cast<float>(atoi(argv[3])));
    else if(atoi(argv[2]) == 4)
    er3 += run<float8e4m3fn>(n, ml_dtypes::float8_internal::numeric_limits_float8_e4m3fn::max(), static_cast<float>(atoi(argv[3])));    
    else 
    er3 += run<int>(n,1.0, static_cast<int>(atoi(argv[3])));

    std::cout << er3 << std::endl;




    return 0;
}