#define TLAPACK_PREFERRED_MATRIX_LEGACY
#include <tlapack/plugins/legacyArray.hpp>
#include <tlapack/plugins/stdvector.hpp>
#include <tlapack/blas/dotu.hpp>

#define MODE1
#define MODE2


// <T>LAPACK
#include <tlapack/blas/syrk.hpp>
#include <tlapack/lapack/gesvd.hpp>
#include <tlapack/blas/trmm.hpp>
#include <tlapack/blas/axpy.hpp>
#include <tlapack/blas/gemm.hpp>
#include <tlapack/blas/trsv.hpp>
#include <tlapack/lapack/geqr2.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/lansy.hpp>
#include <tlapack/lapack/laset.hpp>
#include <tlapack/lapack/ung2r.hpp>
#include <tlapack/lapack/getrf.hpp>
#include <tlapack/plugins/float8_iee_p.hpp>
//#include <../eigen/Eigen>
// C++ headers
#include <chrono>  // for high_resolution_clock
#include <iostream>
#include <memory>
#include <vector>
#include <fstream>
#include <random>


template <typename real_t>
double run(size_t m, size_t n, real_t scale, float cond, int name, bool arithmetic)
{
    using std::size_t;
    using matrix_t = tlapack::LegacyMatrix<real_t>;
    // Functors for creating new matrices
    tlapack::Create<matrix_t> new_matrix;
 
    // Turn it off if m or n are large
    bool verbose = false;

    // Arrays
    std::vector<real_t> tau(n);
    std::vector<float> tau_f(n);
    std::vector<float> scal(n);

    //declare all required matrices
    std::vector<float> FG_;
    auto FG = new_matrix(FG_, m, n);
    std::vector<real_t> A_;
    auto A = new_matrix(A_, m, n);
    std::vector<real_t> R_;
    auto R = new_matrix(R_, n, n);
    std::vector<real_t> Q_;
    auto Q = new_matrix(Q_, m, n);
    std::vector<float> Rf_;
    auto Rf = new_matrix(Rf_, n, n);
    std::vector<float> Qf_;
    auto Qf = new_matrix(Qf_, n, n);

    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            FG(i,j) = float((static_cast<float>(rand())/static_cast<float>(RAND_MAX)));
            A(i,j) = real_t(FG(i,j));
        }
    }

     //now generate n dimensional vector b to solve min ||Ax - b||_2
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

    //okay now we may begin, first take QR of FG
    tlapack::lacpy(tlapack::GENERAL, A, Q);
    tlapack::geqr2(Q, tau, scal);

    //save R
    tlapack::lacpy(tlapack::UPPER_TRIANGLE, Q, R);
    for(int i = 0; i < m; i++){
            for(int j = 0; j < i; j++){
                R(i,j) = 0;
            }
        }

    tlapack::ung2r(Q, tau);

    //now we need to spend some time making the augmented systems
    std::vector<float> A_aug_;
    auto A_aug = new_matrix(A_aug_, 2*m, 2*n);
    for(int i = 0; i < 2*m; i++) {
        for(int j = 0; j < 2*n; j++) {
            if(i < m && j < n)  A_aug(i,j) = i == j ? 1 : 0;
            else if(i >= m && j < n) A_aug(i,j) = A(j,i - m);
            else if(i <m && j >=n) A_aug(i,j) = A(i, j - n);
            else A_aug(i,j) = 0;
        }
    }
    std::vector<float> R_aug_;
    auto R_aug = new_matrix(R_aug_, 2*m, 2*n);
    for(int i = 0; i < 2*m; i++) {
        for(int j = 0; j < 2*n; j++) {
            if(i < m && j < n)  R_aug(i,j) = R(j,i);
            else if(i >= m && j < n) R_aug(i,j) = ((i - m) == j ? 1 : 0);
            else if(i <m && j >=n) R_aug(i,j) = 0;
            else R_aug(i,j) = R(i-m, j-n);
        }
    }
    std::vector<float> Q_aug_;
    auto Q_aug = new_matrix(Q_aug_, 2*m, 2*n);
    for(int i = 0 ; i < 2*m; i++) {
        for(int j = 0; j < 2*n; j++) {
            if(i < m && j < n) Q_aug(i,j) = Q(i,j);
            else Q_aug(i,j) = 0;
        }
    }

    //now construct augmented vectors
    std::vector<float> b_aug_(n);
    tlapack::LegacyVector<float, idx_t> b_aug(n, b_aug_.data());

    std::vector<float> b_aug_copy_(n);
    tlapack::LegacyVector<float, idx_t> b_aug_copy(n, b_aug_copy_.data());

    std::vector<float> r_aug_(n);
    tlapack::LegacyVector<float, idx_t> r_aug(n, r_aug_.data());

    std::vector<float> s_(n);
    tlapack::LegacyVector<float, idx_t> s(n, s_.data());

    std::vector<float> t_(n);
    tlapack::LegacyVector<float, idx_t> t(n, t_.data());

    std::vector<float> s_plus_t_(2*n);
    tlapack::LegacyVector<float, idx_t> s_plus_t(2*n, s_plus_t_.data());




    for(int i = 0; i < 2*n; i++) {
        if(i < n) {
            b_aug[i] = b[i];
            r_aug[i] = b[i];
            b_aug_copy[i] = b[i];
        } else {
            b_aug[i] = 0;
            r_aug[i] = 0;
            b_aug_copy[i] = 0;
        }
    }

    //now we need to get initial solution
    //procedure is Q_aug*trsv(R_aug, flipped_I*(Q_aug^T)*b_aug)
    tlapack::gemv(tlapack::TRANSPOSE, 1.0, Q_aug, r_aug, 0.0, r_aug);
    for(int i = 0; i < m; i++) {
        auto tmp = r[i];
        r[i] = r[m+i];
        r[m+i] = tmp;
    }
    tlapack::trsv(Uplo::Lower, tlapack::NO_TRANS, Diag::NonUnit, R_aug, r_aug);
    tlapack::gemv(tlapack::NO_TRANS, 1.0, Q_aug, r_aug, 0.0, r_aug);

    //now first soln is in r_aug
    //now set r_aug = A*r_aug - b
    tlapack::gemv(tlapack::NO_TRANS, -1.0, FG, r_aug, 1.0, b_aug_copy);
    //now residula is stored in b_aug_copy
    //now we can begin the actual IR
    float res_norm = 1.0;

    int count = 0; //to keep track of number of IR iterations

    do {
        if(count > 10) break;
        count++;



    } while(res_norm > 0.0001)




}

int main(int argc, char** argv) {

    typedef ml_dtypes::float8_e4m3fn float8e4m3fn;
    typedef ml_dtypes::float8_e5m2 float8e5m2;
    typedef ml_dtypes::float8_ieee_p<4> floate4m3;
    typedef ml_dtypes::float8_ieee_p<3> floate5m2;
    typedef ml_dtypes::block_float8_ieee<4> bfp;
    int m, n;


    // Default arguments
    m = (argc < 2) ? 5 : atoi(argv[1]);
    n = (argc < 3) ? 5 : atoi(argv[2]);
    double err1 = 0;
    double er3 = 0;
    double err2 = 0;

    std::ifstream f;

    for (int i = 1; i < 1001; i += 2000){
    srand(atoi(argv[3]) + i);  // Init random seed

    std::cout.precision(10);
    std::cout << std::scientific << std::showpos;

   
    if(atoi(argv[5]) == 0)
    er3 += run<floate4m3>(m, n, ml_dtypes::float8_internal::numeric_limits_float8_ieee_p<4>::max()/floate4m3{2.0}, static_cast<float>(atoi(argv[3])), i, atoi(argv[4]) == 1);    
    else if(atoi(argv[5]) == 1)
    er3 += run<floate5m2>(m, n, ml_dtypes::float8_internal::numeric_limits_float8_ieee_p<3>::max(), static_cast<float>(atoi(argv[3])), i, atoi(argv[4]) == 1);  
    else if(atoi(argv[5]) == 2)
    er3 +=   run<float>(m,n,1.0, static_cast<float>(atoi(argv[3])), i, atoi(argv[4]) == 1);
    else if(atoi(argv[5]) == 3)
    er3 += run<bfp>(m,n,bfp(1000.0), static_cast<float>(atoi(argv[3])), i, atoi(argv[4]) == 1);
    else if(atoi(argv[5]) == 4)
    er3 += run<float8e4m3fn>(m, n, ml_dtypes::float8_internal::numeric_limits_float8_e4m3fn::max(), static_cast<float>(atoi(argv[3])), i, atoi(argv[4]) == 1);    
    else 
    er3 += run<int>(m,n,1.0, static_cast<int>(atoi(argv[3])), i, atoi(argv[4]) == 1);
    }
    
    using matrix_t = tlapack::LegacyMatrix<bfp>;
    std::vector<bfp> A_;
    std::vector<bfp> B_;
    std::vector<bfp> C_;
    tlapack::Create<matrix_t> new_matrix;
    
    

            
          
   std::cout << float(er3) << std::endl;
    return 0;

}