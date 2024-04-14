/// @file example_geqr2.cpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Plugins for <T>LAPACK (must come before <T>LAPACK headers)
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
#include <tlapack/blas/gemm.hpp>
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

//------------------------------------------------------------------------------
/// Print matrix A in the standard output
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

template <typename matrix_t>
void writeMatrix(const matrix_t& A, std::ofstream& file)
{
    using idx_t = tlapack::size_type<matrix_t>;
    const idx_t m = tlapack::nrows(A);
    const idx_t n = tlapack::ncols(A);

    for (idx_t i = 0; i < m; ++i) {
       // file << std::endl;
        for (idx_t j = 0; j < n; ++j)
            file << A(i, j) << ",";
    }
    
}

template <typename matrix_t>
void readMatrix(matrix_t& A, std::ifstream& file)
{
    using idx_t = tlapack::size_type<matrix_t>;
    const idx_t m = tlapack::nrows(A);
    const idx_t n = tlapack::ncols(A);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
           
        }
    }
}



// template <typename matrix_t>
// bool isNanorInf(const matrix_t& A)
// {
//     int m = tlapack::nrows(A);
//     int n = tlapack::ncols(A);
//     bool to_ret = false;
//     for(int i = 0; i < m ; i ++){
//     for(int j = 0 ; j < n; j++) {
//         to_ret = to_ret | isnan(A(i,j)) | isinf(A(i,j));
//     }
//     }
//     return to_ret;
// }

template <typename matrix_t>
bool isZero(const matrix_t& A)
{
    int m = tlapack::nrows(A);
    int n = tlapack::ncols(A);
    bool to_ret = false;
    for(int i = 0; i < m ; i ++){
    for(int j = 0 ; j < n; j++) {
        to_ret = to_ret & A(i,j) == 0;
    }
    }
    return to_ret;
}

//------------------------------------------------------------------------------
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
    std::vector<double> tau_buffer(n);
    std::vector<double> tau_buffer2(n);

    



    float actual_cond =0.0;

   
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




    std::vector<float> Scal_(n,0.0);
    std::vector<float> sums(n,0.0);

    std::vector<float> scal1(n,1.0);
    std::vector<float> scal2(n,1.0);

    // Initialize arrays with junk
    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < m; ++i) {
            A(i, j) = static_cast<float>(0xDEADBEEF);
            Q(i, j) = static_cast<float>(0xCAFED00D);
            Qf(i, j) = static_cast<float>(0xCAFED00D);
            FG(i,j) = float((static_cast<float>(rand())/static_cast<float>(RAND_MAX)));
            //FG(i,j) = sqrt(-2.0*log(FG(i,j)))*cos(2.0*M_PI*static_cast<float>(rand())/static_cast<float>(RAND_MAX));
        }
        for (size_t i = 0; i < n; ++i) {
            R(i, j) = static_cast<float>(0xFEE1DEAD);
            Rf(i, j) = static_cast<float>(0xFEE1DEAD);
        }
        tau[j] = static_cast<float>(0xFFBADD11);
        tau_f[j] = static_cast<float>(0xFFBADD11);
    }
    
    
    
   
    
   
    // Frobenius norm of A
    float normA = tlapack::lange(tlapack::INF_NORM, FG);

    for(int i =0; i < m; i++) {
        for(int j = 0; j <n; j++){
            sums[j] += abs(FG(i,j));
        }
    }
   
    
    for(int k = 0; k < n; k++){
        Scal_[k] = sqrt(float(scale)*0.125)/sums[k];
        //Scal_[k] = sqrt(float(scale)*0.125)/normA;
        //Scal_[k] = 1;
    }
    
   
     for (size_t j = 0; j < n; ++j){
        for (size_t i = 0; i < m; ++i){
            A(i,j) = static_cast<real_t>(FG(i,j)*Scal_[j]);
        }
     }



    // Print A
    if (verbose) {
        std::cout << std::endl << "A = ";
        printMatrix(FG);
    }


    // Copy A to Q
    tlapack::lacpy(tlapack::GENERAL, A, Q);
    tlapack::lacpy(tlapack::GENERAL, FG, Qf);
    //printMatrix(A);

    // 1) Compute A = QR (Stored in the matrix Q)
    
    // Record start time
    auto startQR = std::chrono::high_resolution_clock::now();
    {
        // QR factorization
        
        tlapack::geqr2(Q, tau, scal1);
        tlapack::geqr2(Qf, tau_f, scal2);

        // Save the R matrix
        tlapack::lacpy(tlapack::UPPER_TRIANGLE, Q, R);
        tlapack::lacpy(tlapack::UPPER_TRIANGLE, Qf, Rf);

        // Generates Q = H_1 H_2 ... H_n
        tlapack::ung2r(Q, tau);
        tlapack::ung2r(Qf, tau_f);
        //compute Q in 32 bits
    }
    // Record end time
    auto endQR = std::chrono::high_resolution_clock::now();

    // Compute elapsed time in nanoseconds
    auto elapsedQR =
        std::chrono::duration_cast<std::chrono::nanoseconds>(endQR - startQR);

    // Compute FLOPS
    double flopsQR =
        (4.0e+00 * ((double)m) * ((double)n) * ((double)n) -
         4.0e+00 / 3.0e+00 * ((double)n) * ((double)n) * ((double)n)) /
        (elapsedQR.count() * 1.0e-9);

    // Print Q and R
   

    double norm_orth_1, norm_repres_1;
    double norm_repres_2, norm_repres_3;

    // 2) Compute ||Q'Q - I||_F

    {
        std::vector<real_t> work_;
        auto work = new_matrix(work_, n, n);
        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < n; ++i)
                work(i, j) = static_cast<float>(0xABADBABE);

        // work receives the identity n*n
        tlapack::laset(tlapack::UPPER_TRIANGLE, 0.0, 1.0, work);
        // work receives Q'Q - I
        tlapack::syrk(tlapack::Uplo::Upper, tlapack::Op::Trans, static_cast<real_t>(1.0), Q, static_cast<real_t>(-1.0),
                      work);

        // Compute ||Q'Q - I||_F
        norm_orth_1 =
            double(tlapack::lansy(tlapack::INF_NORM, tlapack::UPPER_TRIANGLE, work));

        if (verbose) {
            std::cout << std::endl << "Q'Q-I = ";
            printMatrix(work);
        }
    }

    // 3) Compute ||QR - A||_F / ||A||_F
     double letscheck;
    {
        std::vector<real_t> work_;
        auto work = new_matrix(work_, m, n);
        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < m; ++i)
                work(i, j) = static_cast<float>(0xABADBABE);

        // Copy Q to work
        tlapack::lacpy(tlapack::GENERAL, Q, work);

        tlapack::trmm(tlapack::Side::Right, tlapack::Uplo::Upper,
                      tlapack::Op::NoTrans, tlapack::Diag::NonUnit, static_cast<real_t>(1.0), R,
                      work);

        
        std::vector<float> FE_;
        auto FE = new_matrix(FE_, m, n);
        std::vector<float> FEf_;
        auto FEf = new_matrix(FE_, m, n);
        std::vector<float> oFEf_;
        auto oFEf = new_matrix(oFEf_, m, n);
        for(int i = 0; i < m; i++){
            for(int j = 0; j < i; j++){
                R(i,j)  =0;
                Rf(i,j) = 0;
            }
        }

        
        // for(int i = 0; i < n; i++)
        // {
        //     scal1[i] = std::pow(2.0,scal1[i]);
        //     scal2[i] = std::pow(2.0,scal2[i]);
        // }



        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < m; ++i){
                FE(i, j) = float(work(i,j))/Scal_[j] - FG(i, j);
                FEf(i, j) = Qf(i,j) - float(Q(i,j));
                oFEf(i, j) = Rf(i,j) - float(R(i,j))/Scal_[j];
          
            }
         if (verbose) {
        std::cout << std::endl << "Q = ";
        printMatrix(Q);
        std::cout << std::endl << "FEf = ";
        printMatrix(FEf);
        std::cout << std::endl << "R = ";
        printMatrix(R);
        std::cout << std::endl << "oFEf = ";
        printMatrix(oFEf);std::cout << std::endl;
    }
    
        double normR = double(tlapack::lange(tlapack::MAX_NORM,Rf));
        
        double normQ = double(tlapack::lange(tlapack::MAX_NORM,Qf));
        norm_repres_1 = double(tlapack::lange(tlapack::INF_NORM, FE))/normA ;
        norm_repres_2 = double(tlapack::lange(tlapack::MAX_NORM, FEf))/normQ;
        norm_repres_3 = double(tlapack::lange(tlapack::MAX_NORM, oFEf))/normR;

        std::cout << normA << std::endl;

        //printMatrix(FE);
        // std::cout << "-------------------------------------------------------" << std::endl;
        // printMatrix(FG);
        // std::cout << "-------------------------------------------------------" << std::endl;


    }

    
 
    return norm_repres_1;

    
}

//------------------------------------------------------------------------------

int main(int argc, char** argv)
{

    typedef ml_dtypes::float8_e4m3fn float8e4m3fn;
    typedef ml_dtypes::float8_e5m2 float8e5m2;
    typedef ml_dtypes::float8_ieee_p<4> floate4m3;
    typedef ml_dtypes::float8_ieee_p<3> floate5m2;
    typedef ml_dtypes::block_float8_ieee<4> bfp4;
    typedef ml_dtypes::block_float8_ieee<3> bfp3;

    int m, n;

    



    // Default arguments
    m = (argc < 2) ? 5 : atoi(argv[1]);
    n = (argc < 3) ? 5 : atoi(argv[2]);
    double err1 = 0;
    double er3 = 0;
    double err2 = 0;

    std::ifstream f;


    for (int i = 1; i < 1001; i += 50){
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
    er3 += run<bfp4>(m,n,bfp4(1000.0), static_cast<float>(atoi(argv[3])), i, atoi(argv[4]) == 1);
    else if(atoi(argv[5]) == 4)
    er3 += run<float8e4m3fn>(m, n, ml_dtypes::float8_internal::numeric_limits_float8_e4m3fn::max(), static_cast<float>(atoi(argv[3])), i, atoi(argv[4]) == 1);    
    else 
    er3 += run<int>(m,n,1.0, static_cast<int>(atoi(argv[3])), i, atoi(argv[4]) == 1);
    }
    
    // using matrix_t = tlapack::LegacyMatrix<bfloat>;
    // std::vector<bfloat> A_;
    // std::vector<bfloat> B_;
    // std::vector<bfloat> C_;
    // tlapack::Create<matrix_t> new_matrix;

    // auto A = new_matrix(A_, 5, 5);
    // auto B = new_matrix(B_, 5, 5);
    // auto C = new_matrix(C_, 5, 5);
    // tlapack::gemm(tlapack::NO_TRANS, tlapack::NO_TRANS, bfloat{1.0},A, B, C);
    

          
   std::cout << float(er3/20.0) << std::endl;
}
