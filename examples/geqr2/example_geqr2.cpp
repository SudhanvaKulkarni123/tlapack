/// @file example_geqr2.cpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
// Plugins for <T>LAPACK (must come before <T>LAPACK headers)
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
//------------------------------------------------------------------------------

template<typename T, typename matrix_t>
int constructMatrix(int m, int n, float cond, int space, bool geom, matrix_t& A, int p, double& da_ref, matrix_t& Ai, matrix_t& ATA) {
    //this is an ambitious function that uses a Python embedding to call the functions found in generate\ copy.py to fill in the entries of A
    
    PyObject *pName, *pModule, *pFunc;
    PyObject *pArgs, *pValue;
    int i;
    setenv("PYTHONPATH", ".", 1);
    Py_Initialize();
    pName = PyUnicode_DecodeFSDefault((char*)"gen");
    pModule = PyImport_Import(pName);
    Py_DECREF(pName);
    if (pModule != NULL) {
        pFunc = PyObject_GetAttrString(pModule, (char *)"LU_gen");

        if (pFunc && PyCallable_Check(pFunc)) {
            pArgs = PyTuple_New(6);
            for (i = 0; i < 6; ++i) {
                switch(i) {
                    case 0:
                        pValue = PyLong_FromLong(m);
                        break;
                    case 1:
                        pValue = PyLong_FromLong(n);
                        break;
                    case 2:
                        pValue = PyFloat_FromDouble(cond);
                        break;
                    case 3:
                        pValue = PyLong_FromLong(space);
                        break;
                    case 4:
                        pValue = geom ? Py_True : Py_False;
                        break;
                    default:
                        pValue = PyLong_FromLong(p);
                        break;
                }
                
                if (!pValue) {
                    Py_DECREF(pArgs);
                    Py_DECREF(pModule);
                    fprintf(stderr, "Cannot convert argument\n");
                    return 1;
                }
                /* pValue reference stolen here: */
                PyTuple_SetItem(pArgs, i, pValue);
            }
            pValue = PyObject_CallObject(pFunc, pArgs);
            Py_DECREF(pArgs);
            if (pValue != NULL) {
                std::cout << PyFloat_AsDouble(PyList_GetItem(pValue, 6)) << std::endl;
                
                for(int j = 0; j < n; j++){
                    for(int i = 0 ; i < m; i++) {
                        A(i,j) = static_cast<float>(static_cast<T>(PyFloat_AsDouble(PyList_GetItem(pValue, m*j + i))));
                    }
                }
                
                
                
                // tlapack::LegacyMatrix<T, int> LU(n, n, b.data(), n);
                // printMatrix(LU);
                std::cout << "condition number: " << PyFloat_AsDouble(PyList_GetItem(pValue, m*n)) << std::endl;
                std::cout << "epsilon: " << std::pow(2.0,-22) << std::endl;
                std::cout << "eps * cond" << PyFloat_AsDouble(PyList_GetItem(pValue, m*n)) * std::pow(2.0,-22) << std::endl;                //get computed condition number
                da_ref = PyFloat_AsDouble(PyList_GetItem(pValue, m*n));
                //reconstruct the psuedoinverse in column major format
                for(int j = 0; j < m; j++){
                    for(int i = 0 ; i < n; i++) {
                        Ai(i,j) = static_cast<float>((PyFloat_AsDouble(PyList_GetItem(pValue, m*n + 1 + n*j + i))));
                    }
                }
                
                //reconstruct A^T*A in column major format
                for(int j = 0; j < n; j++){
                    for(int i = 0 ; i < n; i++) {
                        ATA(i,j) = static_cast<float>((PyFloat_AsDouble(PyList_GetItem(pValue, 2*m*n + 1 + n*j + i))));
                    }
                }
                


                Py_DECREF(pValue);
            }
            else {
                Py_DECREF(pFunc);
                Py_DECREF(pModule);
                PyErr_Print();
                fprintf(stderr,"Call failed\n");
                return 1;
            }
        }
        else {
            if (PyErr_Occurred())
                PyErr_Print();
            fprintf(stderr, "Cannot find function for gen\n");
        }
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    }
    else {
        PyErr_Print();
        fprintf(stderr, "Failed to load program\n");
        return 1;
    }
    if (Py_FinalizeEx() < 0) {
        return 120;
    }
    return 0;
    }






//------------------------------------------------------------------------------
template <typename real_t>
double run(size_t m, size_t n, real_t scale, float cond, int name, bool arithmetic, int p)
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
    std::vector<float> FGi_;
    auto FGi = new_matrix(FGi_, n, m);
    std::vector<float> FGTFG_; 
    auto FGTFG = new_matrix(FGTFG_, n, n);




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
            FG(i,j) = float(static_cast<real_t>((static_cast<float>(rand())/static_cast<float>(RAND_MAX))));
            //FG(i,j) = sqrt(-2.0*log(FG(i,j)))*cos(2.0*M_PI*static_cast<float>(rand())/static_cast<float>(RAND_MAX));
        }
        for (size_t i = 0; i < n; ++i) {
            R(i, j) = static_cast<float>(0xFEE1DEAD);
            Rf(i, j) = static_cast<float>(0xFEE1DEAD);
        }
        tau[j] = static_cast<float>(0xFFBADD11);
        tau_f[j] = static_cast<float>(0xFFBADD11);
    }
    

    double cond_ref;
    constructMatrix<real_t>(m, n, cond, std::ceil(cond/static_cast<float>(5)) > n-1 ? n-1 : std::ceil(cond/static_cast<float>(5)) , true, FG, p, cond_ref, FGi, FGTFG);
   
    
   
    
   
    // Frobenius norm of A
    float normA = tlapack::lange(tlapack::INF_NORM, FG);

 

    for(int i =0; i < m; i++) {
        for(int j = 0; j <n; j++){
            sums[i] += abs(FG(i,j));
        }
    }
   
    
    for(int k = 0; k < n; k++){
        //Scal_[k] = sqrt(float(scale)*0.125)/sums[k];
       Scal_[k] = sqrt(float(scale)*0.125)/normA;
        //Scal_[k] = 1;
    }
   
   
     for (size_t j = 0; j < n; ++j){
        for (size_t i = 0; i < m; ++i){
            A(i,j) = static_cast<real_t>(FG(i,j)*Scal_[j]);
        }
     }

    
    


     



    // Print A
 


    // Copy A to Q
    tlapack::lacpy(tlapack::GENERAL, A, Q);
    tlapack::lacpy(tlapack::GENERAL, FG, Qf);

    

    // 1) Compute A = QR (Stored in the matrix Q)
    
    // Record start time
    auto startQR = std::chrono::high_resolution_clock::now();
    {
        // QR factorization
        
        tlapack::geqr2(Q, tau, scal1);
       

        // Save the R matrix
        tlapack::lacpy(tlapack::UPPER_TRIANGLE, Q, R);

        // Generates Q = H_1 H_2 ... H_n
        tlapack::ung2r(Q, tau);

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



        for (size_t j = 0; j < n; ++j){
            for (size_t i = 0; i < m; ++i){
                FE(i, j) = float(work(i,j))/Scal_[j] - FG(i, j);
          
            }
        }


    
        norm_repres_1 = double(tlapack::lange(tlapack::INF_NORM, FE))/normA ;

}

    

    
 
    return norm_repres_1;

    
}


//------------------------------------------------------------------------------

int main(int argc, char** argv)
{

    std::ofstream myfile("dots.csv");
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


   
    srand(atoi(argv[3]));  // Init random seed

    std::cout.precision(10);
    std::cout << std::scientific << std::showpos;

   
    if(atoi(argv[5]) == 0)
    er3 = run<floate4m3>(m, n, ml_dtypes::float8_internal::numeric_limits_float8_ieee_p<4>::max()/floate4m3{2.0}, static_cast<float>(atoi(argv[3])), 0, atoi(argv[4]) == 1, 4);    
    else if(atoi(argv[5]) == 1)
    er3 = run<floate5m2>(m, n, ml_dtypes::float8_internal::numeric_limits_float8_ieee_p<3>::max(), static_cast<float>(atoi(argv[3])), 0, atoi(argv[4]) == 1, 3);  
    else if(atoi(argv[5]) == 2)
    er3 =   run<float>(m,n,1.0, static_cast<float>(atoi(argv[3])), 0, atoi(argv[4]) == 1, 4);
    else if(atoi(argv[5]) == 3)
    er3 = run<bfp4>(m,n,bfp4(1000.0), static_cast<float>(atoi(argv[3])), 0, atoi(argv[4]) == 1, 4);
    else if(atoi(argv[5]) == 4)
    er3 = run<float8e4m3fn>(m, n, ml_dtypes::float8_internal::numeric_limits_float8_e4m3fn::max(), static_cast<float>(atoi(argv[3])), 0, atoi(argv[4]) == 1, 4);    
    else 
    er3 = run<int>(m,n,1.0, static_cast<int>(atoi(argv[3])), 0, atoi(argv[4]) == 1, 4);
    std::cout << float(er3) << "," << std::endl;
    
    
    // using matrix_t = tlapack::LegacyMatrix<bfloat>;
    // std::vector<bfloat> A_;
    // std::vector<bfloat> B_;
    // std::vector<bfloat> C_;
    // tlapack::Create<matrix_t> new_matrix;

    // auto A = new_matrix(A_, 5, 5);
    // auto B = new_matrix(B_, 5, 5);
    // auto C = new_matrix(C_, 5, 5);
    // tlapack::gemm(tlapack::NO_TRANS, tlapack::NO_TRANS, bfloat{1.0},A, B, C);
    

          
   
}
