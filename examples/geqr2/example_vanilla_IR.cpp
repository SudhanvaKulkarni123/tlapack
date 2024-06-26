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

template <typename matrix_t>
void printMatrix(const matrix_t& A)
{
    using idx_t = tlapack::size_type<matrix_t>;
    const idx_t m = tlapack::nrows(A);
    const idx_t n = tlapack::ncols(A);

    for (idx_t i = 0; i < m; ++i) {
        std::cout << std::endl;
        for (idx_t j = 0; j < n; ++j)
            std::cout << A(i, j) << ",";
    }
    
}

template <typename real_t>
double run(size_t m, size_t n, real_t scale, float cond, int name, bool arithmetic)
{
    using std::size_t;
    using matrix_t = tlapack::LegacyMatrix<real_t>;
    using idx_t = int;
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
    auto R = new_matrix(R_, m, n);
    std::vector<float> R_copy_;
    auto R_copy = new_matrix(R_copy_, n, n);
    std::vector<real_t> Q_;
    auto Q = new_matrix(Q_, m, m);
    std::vector<float> Q_copy_;
    auto Q_copy = new_matrix(Q_copy_, m, m);
    std::vector<float> Rf_;
    auto Rf = new_matrix(Rf_, m, n);
    std::vector<float> Qf_;
    auto Qf = new_matrix(Qf_, m, m);
    std::vector<float> Q1_;
    auto Q1 = new_matrix(Qf_, m, n);
    std::vector<float> Q2_;
    auto Q2 = new_matrix(Q2_, m, m-n);

    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            FG(i,j) = float((static_cast<float>(rand())/static_cast<float>(RAND_MAX)));
            A(i,j) = real_t(FG(i,j));
        }
    }

     //now generate n dimensional vector b to solve min ||Ax - b||_2
    std::vector<float> b_(m);
    tlapack::LegacyVector<float, idx_t> b(m, b_.data());
    //copy of b
    std::vector<float> b1_(m);
    tlapack::LegacyVector<float, idx_t> b1(m, b1_.data());
    //residual
    std::vector<float> r_(m);
    tlapack::LegacyVector<float, idx_t> r(n, r_.data());
    //residual copy
    std::vector<float> r_copy_(m);
    tlapack::LegacyVector<float, idx_t> r_copy(n, r_.data());
    //buffer vectors
    std::vector<float> s_(m);
    tlapack::LegacyVector<float, idx_t> s(m, s_.data());

    std::vector<float> t_(n);
    tlapack::LegacyVector<float, idx_t> t(n, t_.data());

    std::vector<float> s_and_t_(n+m);
    tlapack::LegacyVector<float, idx_t> s_and_t(m + n, s_and_t_.data());

    std::vector<float> u_(m);
    tlapack::LegacyVector<float, idx_t> u(m, u_.data());

    std::vector<float> u_buf_(m);
    tlapack::LegacyVector<float, idx_t> u_buf(m, u_.data());

    std::vector<float> v_(n);
    tlapack::LegacyVector<float, idx_t> v(n, v_.data());

    std::vector<float> c_(n);
    tlapack::LegacyVector<float, idx_t> c(n, c_.data());

    std::vector<float> d_(m - n);
    tlapack::LegacyVector<float, idx_t> d(m - n, d_.data());

    std::vector<float> e_(n);
    tlapack::LegacyVector<float, idx_t> e(n, d_.data());

    //soln vector
    std::vector<float> x_(n);
    tlapack::LegacyVector<float, idx_t> x(n, x_.data());


    for( int i = 0; i < m; i++) {
        b[i] = (static_cast<float>(rand()))/static_cast<float>(RAND_MAX);
        s[i] = b[i];
        b1[i] = b[i];
        r[i] = b[i];
        r_copy[i] = b[i];
        s_and_t[i] = s[i];
    }

    for(int i = 0; i < n; i++) {
        t[i] = 0;
        s_and_t[m + i] = t[i];

    }

    //okay now we may begin, first take QR of FG
    tlapack::lacpy(tlapack::GENERAL, A, Q);
    tlapack::geqr2(Q, tau, scal);

    tlapack::lacpy(tlapack::GENERAL, FG, Qf);
    tlapack::geqr2(Qf, tau_f, scal);

    //save R
    tlapack::lacpy(tlapack::UPPER_TRIANGLE, Q, R);
    tlapack::lacpy(tlapack::UPPER_TRIANGLE, Qf, Rf);
    for(int i = 0; i < m; i++){
            for(int j = 0; j < i; j++){
                R(i,j) = static_cast<real_t>(0.0);
                Rf(i,j) = 0;
            }
        }
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            R_copy(i,j) = static_cast<float>(R(i,j));
        }
    }
    tlapack::ung2r(Q, tau);
    tlapack::ung2r(Qf, tau_f);

    //now we need to spend some time making the augmented system
    std::vector<float> A_aug_;
    auto A_aug = new_matrix(A_aug_, 2*m, 2*n);
    for(int i = 0; i < 2*m; i++) {
        for(int j = 0; j < 2*n; j++) {
            if(i < m && j < n)  A_aug(i,j) = i == j ? 1 : 0;
            else if(i >= m && j < n) A_aug(i,j) = static_cast<float>(A(j,i - m));
            else if(i <m && j >=n) A_aug(i,j) = static_cast<float>(A(i, j - n));
            else A_aug(i,j) = 0;
        }
    }

    int cnt = 0;
    auto r_norm = 0.0;
    auto dr_norm = 0.0;
    auto old_dr_norm = 0.0;

    //now split Q
    for(int i =0; i < m; i++) {
        for(int j = 0; j< n; j++) {
            Q1(i,j) = static_cast<float>(Q(i,j));
        }
    }

    for(int i = n; i < m; i++) {
        for(int j  =0; j < m - n; j++) {
            Q2(i,j) = static_cast<float>(Q(i,n + j));
        }
    }

    tlapack::gemv(tlapack::TRANSPOSE, 1.0, Q1, s, 0.0, c);          //c = Q1^Ts
    tlapack::gemv(tlapack::TRANSPOSE, 1.0, Q2, s, 0.0, d);          //d= Q2^Ts
    tlapack::trsv(Uplo::Lower, tlapack::TRANSPOSE, Diag::NonUnit, R_copy, t);       //e is now stored in t
    for(int i = 0; i < n; i++) {
         v[i] = -t[i];      //now v = -e
         e[i] = t[i];
    }
    tlapack::axpy(1.0, c, v);   //now v = inv(R)*(c-v)
    tlapack::trsv(Uplo::Upper, tlapack::NO_TRANS, Diag::NonUnit, R_copy, v);
    for (int i = 0; i < n; i++) {
        if(i < m) u_buf[i] = c[i];
        else u_buf[i+m] = d[i];
    }
    tlapack::gemv(tlapack::NO_TRANS,1.0,Q_copy,u_buf,0.0,u);

    //initial soln is u,v
    for(int i =0; i < m; i++) {
        r[i] = u[i];
        r_copy[i] = u[i];
    }
    for(int i =0; i < n; i++) {
        x[i] = v[i];
    }
    do {
        tlapack::axpy(-1.0, b, r_copy);
        for(int i = 0; i < m; i++) {
            s[i] = r_copy[i];
        }
        tlapack::gemv(tlapack::NO_TRANS, 1.0, FG, x, 1.0, s);
        for(int i = 0; i < m; i++) {
            s[i] = -s[i];
        }

        tlapack::gemv(tlapack::TRANSPOSE, -1.0, FG, r, 0.0, t);

        //repeat soln procedure
        tlapack::gemv(tlapack::TRANSPOSE, 1.0, Q1, s, 0.0, c);          //c = Q1^Ts
        tlapack::gemv(tlapack::TRANSPOSE, 1.0, Q2, s, 0.0, d);          //d= Q2^Ts
        tlapack::trsv(Uplo::Lower, tlapack::TRANSPOSE, Diag::NonUnit, R_copy, t);       //e is now stored in t
        for(int i = 0; i < n; i++) {
            v[i] = -t[i];      //now v = -e
            e[i] = t[i];
        }
        tlapack::axpy(1.0, c, v);   //now v = inv(R)*(c-v)
        tlapack::trsv(Uplo::Upper, tlapack::NO_TRANS,Diag::NonUnit, R_copy, v);
        for (int i = 0; i < n; i++) {
            if(i < m) u_buf[i] = c[i];
            else u_buf[i+m] = d[i];
        }
        tlapack::gemv(tlapack::NO_TRANS,1.0,Q_copy,u_buf,0.0,u);
        //update soln vector
        tlapack::axpy(1.0, u, r);
        tlapack::axpy(1.0, v, x);
        for(int i = 0; i < m; i++) {
            r_copy[i] = r[i];
        }
        cnt++;
        for(int i = 0; i < m; i++) {
            r_norm = r[i]*r[i];
            dr_norm = u[i]*u[i];
        }



    } while (sqrt(r_norm) > 0.001);
   







    
    //now post-process
    return 0.0;



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