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
        file << std::endl;
        for (idx_t j = 0; j < n; ++j)
            file << A(i, j) << ",";
    }
    
}

template <typename matrix_t>
bool isNanorInf(const matrix_t& A)
{
    int m = tlapack::nrows(A);
    int n = tlapack::ncols(A);
    bool to_ret = false;
    for(int i = 0; i < m ; i ++){
    for(int j = 0 ; j < n; j++) {
        to_ret = to_ret | isnan(A(i,j)) | isinf(A(i,j));
    }
    }
    return to_ret;
}

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
std::ofstream myfile("../e5m2_error_e_cond.csv");
template <typename real_t>
double run(size_t m, size_t n, real_t scale, float cond, int name, bool arithmetic)
{
    using std::size_t;
    using matrix_t = tlapack::LegacyMatrix<real_t>;
    using matrix_ft = tlapack::LegacyMatrix<float>;
    using matrix_dt = tlapack::LegacyMatrix<double>;

    //file to store matrix for condition number computation
    std::ofstream matfile("./mat/" + std::to_string(name) + ".txt");
    matfile << std::scientific;
    matfile << n ;

    std::ofstream fmatfile("./fmat/" + std::to_string(name) + ".txt");
    fmatfile << std::scientific;
    fmatfile << n ;


    // Functors for creating new matrices
    tlapack::Create<matrix_t> new_matrix;
    tlapack::Create<matrix_ft> new_fmatrix;
    tlapack::Create<matrix_dt> new_dmatrix;

    // Turn it off if m or n are large
    bool verbose = false;

    // Arrays
    std::vector<real_t> tau(n);
    std::vector<float> tau_f(n);
    std::vector<double> tau_buffer(n);
    std::vector<double> tau_buffer2(n);

    std::random_device generator;
    std::normal_distribution<double> distribution(0.0,1.0);
    // Matrices
    std::vector<double> R1_;
    auto R1 = new_dmatrix(R1_, m, n);
    std::vector<double> R2_;
    auto R2 = new_dmatrix(R2_, m, n);
    for(int j = 0; j < n; ++j){
        for(int i = 0; i < m; ++i){
          
            // R1(i,j) = (static_cast<double>(rand()))/static_cast<double>(RAND_MAX);
            // R2(i,j) = (static_cast<double>(rand()))/static_cast<double>(RAND_MAX);

            R1(i,j) = distribution(generator);
            R2(i,j) = distribution(generator);
            
        }
    }
    


    







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
    std::vector<float> iA_;
    auto iA = new_matrix(iA_, m, n);

    std::vector<float> S1_;
    auto S1 = new_fmatrix(S1_,m,n);
    std::vector<float> S2_;
    auto S2 = new_fmatrix(S1_,m,n);

    std::vector<float> Scal_(n,0.0);
    std::vector<float> sums(n,0.0);

    // Initialize arrays with junk
    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < m; ++i) {
            A(i, j) = static_cast<float>(0xDEADBEEF);
            Q(i, j) = static_cast<float>(0xCAFED00D);
            Qf(i, j) = static_cast<float>(0xCAFED00D);
        }
        for (size_t i = 0; i < n; ++i) {
            R(i, j) = static_cast<float>(0xFEE1DEAD);
            Rf(i, j) = static_cast<float>(0xFEE1DEAD);
        }
        tau[j] = static_cast<float>(0xFFBADD11);
        tau_f[j] = static_cast<float>(0xFFBADD11);
    }

    
    
    
    std::vector<double> iS_(n*m, 0.0);
    auto iS = new_matrix(iS_, m, n);
    std::vector<double> iR_(n*m, 0.0);
    auto iR = new_matrix(iR_, m, n);
 
    tlapack::geqr2(R1, tau_buffer);
    tlapack::ung2r(R1, tau_buffer);
   tlapack::geqr2(R2, tau_buffer);
    tlapack::ung2r(R2, tau_buffer);
    


    

    for(int i = 0; i < n; i++){
        if(arithmetic)
        iS(i,i) = float(1.0 - float(i)*(1.0 - 1.0/cond)/float(n-1));    //-- this is for a linear distribution of singular values
        else
        iS(i,i) = float(std::pow(cond,float(-i)/float(n - 1)));       //-- this is for exponential
        
        
        

    }

   

  
    //now call gemm
    #ifdef MODE1
    tlapack::gemm(tlapack::NO_TRANS,tlapack::NO_TRANS,1.0, iS, R1, 0.0, iA);
    tlapack::gemm(tlapack::NO_TRANS,tlapack::NO_TRANS,1.0, R2, iA, 0.0, FG);
    writeMatrix(FG, fmatfile);
   
    std::vector<float> FG_d;
    std::vector<float> s_dup(n, 0.0);
    auto FG_dup = new_matrix(FG_d, m,n);
    for(int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            FG_dup(i,j) = static_cast<float>(static_cast<real_t>((FG(i,j))));
        }
    }

    writeMatrix(FG_dup, matfile);
    std::vector<float> AG_d;
    auto AG_dup = new_fmatrix(AG_d, m,n);
    std::vector<float> BG_d;
    auto BG_dup = new_fmatrix(BG_d, m,n);
    
   
    float maxdiff = 0.0;
    float max_singular = s_dup[0];
    float min_singular = s_dup[n-1];
    int count;
  
    #else
    //now need to declare a bunch of FP8 matrices and multiply in FP8
    std::vector<float> R1_st;
    auto R1_mat = new_matrix(R1_st, m,n);
    std::vector<float> R2_st;
    auto R2_mat = new_matrix(R2_st, m,n);
    std::vector<float> iS_st;
    auto iS_mat = new_matrix(iS_st, m,n);
    std::vector<float> iA_st;
    auto iA_mat = new_matrix(iA_st, m,n);
    std::vector<float> FG_st;
    auto FG_mat = new_matrix(FG_st, m,n);
   
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            R1_mat(i,j) = static_cast<float>(static_cast<real_t>(R1(i,j)));
            R2_mat(i,j) = static_cast<float>(static_cast<real_t>(R2(i,j)));
            iS_mat(i,j) = static_cast<float>(static_cast<real_t>(iS(i,j)));
            iA_mat(i,j) = static_cast<float>(static_cast<real_t>(iA(i,j)));
        }
    }

    std::vector<float> FG_d;
    std::vector<float> s_dup(n, 0.0);
    auto FG_dup = new_matrix(FG_d, m,n);
   
    tlapack::gemm(tlapack::NO_TRANS,tlapack::NO_TRANS,1.0, iS_mat, R1_mat,0, iA_mat);
    tlapack::gemm(tlapack::NO_TRANS,tlapack::NO_TRANS,1.0, R2_mat, iA_mat,0, FG_mat);
    for(int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            FG(i,j) = static_cast<float>(FG_mat(i,j));
            FG_dup(i,j) = FG(i,j); 
        }
    }
    for(int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            FG_dup(i,j) = static_cast<float>(static_cast<real_t>((FG(i,j))));
        }
    }

    writeMatrix(FG_dup, matfile);


    
    




    




    for(int i = 0; i < m; i++){
        for( int j = 0; j < n; j++){
            sums[j] += abs(FG(i,j));
        }
    }
    #endif

    

    
    //once that's done, call gemm and use the new matrix

    

    
    
   
    // Frobenius norm of A
    float normA = tlapack::lange(tlapack::INF_NORM, FG);
   
    
    for(int k = 0; k < n; k++){
        //Scal_[k] = sqrt(float(scale)*0.125)/sums[k];
        Scal_[k] = sqrt(float(scale)*0.125)/normA;
        //Scal_[k] = 1;
    }
    
    //std::cout << normA;
     for (size_t j = 0; j < n; ++j){
        for (size_t i = 0; i < m; ++i){
            A(i,j) = static_cast<real_t>(FG(i,j)*Scal_[j]);
        }
     }
      if(isNanorInf(FG)) { std::cout << "FG is NAn or Inf" << std::endl;}
    if(isZero(FG)) { std::cout << "FG is zero" << std::endl;}
     if(isNanorInf(A)) { std::cout << "A is NAn or Inf" << std::endl;}
    if(isZero(A)) { std::cout << "A is zero" << std::endl;}

    // Print A
    if (verbose) {
        std::cout << std::endl << "A = ";
        printMatrix(FG);
    }

    // Copy A to Q
    tlapack::lacpy(tlapack::GENERAL, A, Q);
    tlapack::lacpy(tlapack::GENERAL, FG, Qf);

    // 1) Compute A = QR (Stored in the matrix Q)

    // Record start time
    auto startQR = std::chrono::high_resolution_clock::now();
    {
        // QR factorization
        tlapack::geqr2(Q, tau);
         tlapack::geqr2(Qf, tau_f);

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
    if(isNanorInf(work)) { std::cout << "work is NAn or Inf" << std::endl;}
    if(isZero(work)) { std::cout << "work is zero" << std::endl;}
        
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
        //printMatrix(oFEf);
        // std::cout << std::endl << "err = ";
        // printMatrix(FE);

        //check normalization
    }

    
 
    myfile << std::scientific;
    myfile << cond << "," << norm_repres_1 << "," << norm_repres_2 << "," << norm_repres_3 <<  "," << iS(0,0) << "," << iS(n-1,n-1) << "\n";
    return norm_repres_1;

    
}

//------------------------------------------------------------------------------

int main(int argc, char** argv)
{

    typedef ml_dtypes::float8_e4m3fn float8e4m3fn;
    typedef ml_dtypes::float8_e5m2 float8e5m2;
    typedef ml_dtypes::float8_ieee_p<4> floate4m3;
    typedef ml_dtypes::float8_ieee_p<3> floate5m2;
    typedef ml_dtypes::block_float8_ieee<4> bfp;
    int m, n;
    std::ofstream recip_file1("./recip_e4m3greater.csv");
    std::ofstream recip_file2("./recip_e5m2greater.csv");
    std::ofstream recip_file3("./recip_e4m3lesser.csv");
    std::ofstream recip_file4("./recip_e5m2lesser.csv");

    // Default arguments
    m = (argc < 2) ? 5 : atoi(argv[1]);
    n = (argc < 3) ? 5 : atoi(argv[2]);
    double err1 = 0;
    double er3 = 0;
    double err2 = 0;

    for (int i = 1; i < 1001; i += 10){
    srand(i);  // Init random seed

    std::cout.precision(10);
    std::cout << std::scientific << std::showpos;

    // std::cout << bfp{float{5000000000000000005.0}} << std::endl;
    // std::cout << "err : " <<  (float(bfp{float{50000000000000000005.0}}) - float{5000000000000000005.0})/float{5000000000000000005.0} << std::endl;
    

    //  printf("run< float8e4m3fn, L >( %d )\n", n);
    //  std::cout << "epsilon" << ml_dtypes::float8_internal::numeric_limits_float8_e4m3fn::epsilon() << std::endl;
    //run<Eigen::bfloat16>(m,n, 1.0, static_cast<float>(i));
    if(atoi(argv[5]) == 0)
    err1 += run<floate4m3>(m, n, ml_dtypes::float8_internal::numeric_limits_float8_ieee_p<4>::max(), static_cast<float>(atoi(argv[3])), i, atoi(argv[4]) == 1);    
    else if(atoi(argv[5]) == 1)
    err2 += run<floate5m2>(m, n, ml_dtypes::float8_internal::numeric_limits_float8_ieee_p<3>::max(), static_cast<float>(atoi(argv[3])), i, atoi(argv[4]) == 1);  
    else if(atoi(argv[5]) == 2)
    er3 +=   run<float>(m,n,1.0, static_cast<float>(atoi(argv[3])), i, atoi(argv[4]) == 1);
    else if(atoi(argv[5]) == 3)
    er3 += run<bfp>(m,n,bfp(1.0), static_cast<float>(atoi(argv[3])), i, atoi(argv[4]) == 1);
    else 
    er3 += run<int>(m,n,1.0, static_cast<int>(atoi(argv[3])), i, atoi(argv[4]) == 1);
    }
    
            
            
            //if(float(bal1*(floate5m2{1.0}/bal2)) - 1.0 != 0.0) std::cout << bal1 << << "," << i << std::endl;
            

    /*
    generate QA by SLARGE LAPACK routine 
    try to choose random householder in 8-bit
    */
            
        
    

    //run<Eigen::half>(m,n,Eigen::half{1});
    // printf("-----------------------\n");

    // printf("run< float  >( %d, %d )", m, n);

    // er3 += run<float>(m, n, 1.0, 100.0);
    // printf("-----------------------\n");

    // printf("run< double >( %d, %d )", m, n);
    // run<double>(m, n);
    // printf("-----------------------\n");

    // printf("run< long double >( %d, %d )", m, n);
    // run<long double>(m, n);
    // printf("-----------------------\n");
    // std::cout << "e4m3" << std::endl;
    // std::cout << err1 << std::endl;
    // std::cout << "e5m2" << std::endl;
    // std::cout << err2 << std::endl;
    // std::cout << "float32" << std::endl;
    // std::cout << er3 << std::endl;
    //std::cout << er3 << std::endl;
    //std::cout << float(ml_dtypes::float8_internal::numeric_limits_float8_e4m3fn::infinity()) <<std::endl;
    
    std::cout << float(er3)/100.0 << std::endl;
    return 0;
}
