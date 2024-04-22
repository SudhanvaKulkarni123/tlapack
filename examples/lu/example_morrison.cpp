#define TLAPACK_PREFERRED_MATRIX_LEGACY
#include <tlapack/plugins/legacyArray.hpp>
#include <tlapack/plugins/stdvector.hpp>
#include <tlapack/plugins/float8_iee_p.hpp>

// <T>LAPACK
#include <tlapack/blas/trsm.hpp>
#include <tlapack/blas/gemv.hpp>
#include <tlapack/blas/dotu.hpp>
#include <tlapack/blas/trsv.hpp>
#include <tlapack/blas/axpy.hpp>
#include <tlapack/blas/trmm.hpp>
#include <tlapack/lapack/getrf.hpp>
#include <tlapack/lapack/geqr2.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include "../../eigen/Eigen/Core"
#include <tlapack/lapack/ung2r.hpp>
#include <tlapack/blas/scal.hpp>


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

template<typename vec>
void printVector(const vec& v, int m)
{
    for (int i = 0; i < m; ++i) {
        std::cout << v[i] << "," ;
    }
    std::cout << std::endl;
}
//------------------------------------------------------------------------------
template <typename matrix_t, typename T>
T max_elem(matrix_t& A)
{
    using idx_t = tlapack::size_type<matrix_t>;
    const idx_t m = tlapack::nrows(A);
    const idx_t n = tlapack::ncols(A);

    T to_ret = -std::numeric_limits<T>::infinity();
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            to_ret = A(i,j) > to_ret ? A(i,j) : to_ret;
        }
    }

    return to_ret;

}

template <typename matrix_t, typename T>
T min_elem(matrix_t& A)
{
    using idx_t = tlapack::size_type<matrix_t>;
    const idx_t m = tlapack::nrows(A);
    const idx_t n = tlapack::ncols(A);

    T to_ret = std::numeric_limits<T>::infinity();
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            to_ret = A(i,j) < to_ret ? A(i,j) : to_ret;
        }
    }

    return to_ret;

}

template<typename vec>
float find_max(vec& X, int n) {
    float max = 0.0;
    for (int i = 0; i < n; i++) {
        max = abs(X[i]) > max ? abs(X[i]) : max;
    }
    return max;
}

std::ofstream myfile("e5m2_error_f_cond.csv");
template <class T>
float run(size_t n, T scale, float cond) {
    //this function will generate a random b, random A and solve Ax = b using sherman morrison approach
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
            FG(i,j) = (static_cast<float>(rand()))*float(scale)/static_cast<float>(RAND_MAX);
        }
    }


    //now generate n dimensional vector b to solve Ax = b

    std::vector<float> b_(n);
    tlapack::LegacyVector<float, idx_t> b(n, b_.data());

    std::vector<float> b_copy1(n);
    tlapack::LegacyVector<float, idx_t> b1(n, b_copy1.data());

    std::vector<float> b_copy2(n);
    tlapack::LegacyVector<float, idx_t> b2(n, b_copy2.data());

    std::vector<float> b_copy3(n);
    tlapack::LegacyVector<float, idx_t> b3(n, b_copy3.data());

    std::vector<float> e_(n);
    tlapack::LegacyVector<float, idx_t> e(n, e_.data());

    std::vector<float> e_p_(n);
    tlapack::LegacyVector<float, idx_t> e_pure(n, e_p_.data());

    std::vector<float> e_beta_(n);
    tlapack::LegacyVector<float, idx_t> e_beta(n, e_beta_.data());

    for( int i = 0; i < n; i++) {
        b[i] = (static_cast<float>(rand()))/static_cast<float>(RAND_MAX);
        b1[i] = b[i];
        b2[i] = b[i];
        e[i] = 1.0;
        e_pure[i] = 1.0;
    }

    // std::cout << "vector is " << std::endl;
    // printVector(b, n);
    

    //create n-by-n matrix C that will be used to store the result of y = mx + b scaling
    std::vector<T> C_(n * n);
    tlapack::LegacyMatrix<T, idx_t> C(n, n, C_.data(), n);


    // c_ij = z_2*(a_ij - a_min)/(a_max - a_min) + z_1*(a_max - a_ij)/(a_max - a_min)
    float z_1 = -16.0;
    float z_2 = 16.0;

    float a_max = max_elem<tlapack::LegacyMatrix<float, idx_t>,float>(FG);
    float a_min = min_elem<tlapack::LegacyMatrix<float, idx_t> ,float>(FG);
    float div = a_max - a_min;

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            C(i,j) = static_cast<T>(z_2*(FG(i,j) - a_min)/(div) + z_1*(a_max - FG(i,j))/(div));
        }
    }

    float alpha = (z_2 - z_1)/div;
    float beta = (z_1*a_max - z_2*a_min)/div;

    for(int i = 0; i < n; i++) {
        e_beta[i] = -beta;
    }


    //we don't store the B-matrix since it is just a matrix of 1's. i.e B = uu^T where u = (1,1,...,1)^T


    //declare "LU" matrices to actually store the factorization
    std::vector<T> LU_(n * n);
    tlapack::LegacyMatrix<T, idx_t> LU(n, n, LU_.data(), n);

    std::vector<float> LU_float_(n * n);
    tlapack::LegacyMatrix<float, idx_t> LU_float(n, n, LU_float_.data(), n);

   
    // Matrix A is kept unchanged
    tlapack::lacpy(tlapack::GENERAL, C, LU);
    tlapack::lacpy(tlapack::GENERAL, FG, LU_float);

    //declare arrays for piv
    std::vector<size_t> piv_lo(n);
    std::vector<size_t> piv_hi(n);

    //first take LU of C

    int info = tlapack::getrf(LU, piv_lo);
    if (info != 0) {
        std::cerr << "Matrix could not be factorized :(" << std::endl;
        return -1;
    }
    //std::cout << "piv_lo is:" <<  piv_lo[0] << "," << piv_lo[1] << "," << piv_lo[2] << "," << piv_lo[3] << "," << piv_lo[4] << "," << std::endl;
    int infotoo = tlapack::getrf(LU_float, piv_hi);
    if (infotoo != 0) {
        std::cerr << "Matrix could not be factorized in fp32 :(" << std::endl;
        return -1;
    }
    //std::cout << "piv_hi is:" <<  piv_hi[0] << "," << piv_hi[1] << "," << piv_hi[2] << "," << piv_hi[3] << "," << piv_hi[4] << "," << std::endl;

    //create copy to stor low prec LU in high prec
    std::vector<float> LU_copy_(n * n);
    tlapack::LegacyMatrix<float, idx_t> LU_copy(n, n, LU_copy_.data(), n);

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            LU_copy(i,j) = static_cast<float>(LU(i,j));
        }
    }

    tlapack::scal(alpha, b1);
    // std::cout <<"alpha : " << alpha << std::endl;

    for (idx_t i = 0; i < n;i++){
        if (piv_lo[i] != i) {
            auto tmp = b1[piv_lo[i]];
            b1[piv_lo[i]] = b1[i];
            b1[i] = tmp;
        }
    }
    tlapack::trsv(Uplo::Lower, NO_TRANS, Diag::Unit, LU_copy, b1);
    tlapack::trsv(Uplo::Upper, NO_TRANS, Diag::NonUnit, LU_copy, b1);

    

    //okay now that we have the intermediate solution, we can apply Sherman Morrison in fp32 :)
    //For this we first need to compute inv(A)*(1,1,..,1)^T


   
    tlapack::trsv(Uplo::Lower, NO_TRANS, Diag::Unit, LU_copy, e);
    tlapack::trsv(Uplo::Upper, NO_TRANS, Diag::NonUnit, LU_copy, e);

    //to recap, e = inv(A)*u, b_copy1 = inv(A)*b

    //now Sherman Morison
    tlapack::scal(tlapack::dotu(e_beta,b1)/(1 + tlapack::dotu(e_beta,e)), e);
    tlapack::axpy(-1.0,e,b1);


   

    //now find the 32 bit sol
    for (idx_t i = 0; i < n;i++) {
        if (piv_hi[i] != i) {
            auto tmp = b2[piv_hi[i]];
            b2[piv_hi[i]] = b2[i];
            b2[i] = tmp;
        }
    }

    tlapack::trsv(Uplo::Lower, NO_TRANS, Diag::Unit, LU_float, b2);
    tlapack::trsv(Uplo::Upper, NO_TRANS, Diag::NonUnit, LU_float, b2);

    //return norm of difference b/w b1 and b2x

    tlapack::gemv(tlapack::NO_TRANS, 1.0,FG,b1, 0.0, b1);
    // for (idx_t i = n; i-- > 0;) {
    //     if (piv_hi[i] != i) {
    //         auto tmp = b1[piv_hi[i]];
    //         b1[piv_hi[i]] = b1[i];
    //         b1[i] = tmp;
    //     }
    // }
    std::cout << "result :" << b1[0] << "," << b1[1] << "," << b1[2] << "," << b1[3] << "," << b1[4] << "," << std::endl;
    tlapack::gemv(tlapack::NO_TRANS, 1.0,FG,b2, 0.0, b2);
    std::cout << "other result :" << b2[0] << "," << b2[1] << "," << b2[2] << "," << b2[3] << "," << b2[4] << "," << std::endl;
    
    auto nrmb2 = tlapack::nrm2(b2);
    //auto nrmb2 = find_max(b2, n);
    tlapack::axpy(-1.0,b1,b2);
    for(int i = 0; i < n; i++) {
        if(isnan(b2[i])) std::cout << "found nan" << std::endl;
    }

    //||Ax - b||/||A||.||x||
    //return find_max(b2, n)/nrmb2;
    return tlapack::nrm2(b2)/nrmb2;



    //design space - do we want rank 1 update?
    //              - round z1,z2,a_min, a_max to powers of 2?


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