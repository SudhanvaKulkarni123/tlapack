///@author Sudhanva Kulkarni, UC Berkeley
//this file contains code for testing GMRES-IR using 8-bit LU decomposition as a preconditioner
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define TLAPACK_PREFERRED_MATRIX_LEGACY
#include <tlapack/plugins/legacyArray.hpp>
#include <tlapack/plugins/stdvector.hpp>
#include <tlapack/plugins/float8_iee_p.hpp>
#include "tlapack.hpp"
#include "tlapack/blas/gemv.hpp"
#include "tlapack/blas/nrm2.hpp"
#include "tlapack/blas/scal.hpp"
#include "tlapack/blas/axpy.hpp"
#include "tlapack/blas/dot.hpp"
#include "tlapack/blas/iamax.hpp"
#include "tlapack/blas/swap.hpp"
#include "tlapack/blas/trsv.hpp"
#include "tlapack/blas/trsm.hpp"
#include "tlapack/lapack/larfg.hpp"
#include "tlapack/lapack/larft.hpp"
#include "tlapack/lapack/larfb.hpp"
#include "tlapack/lapack/larft.hpp"
#include "tlapack/lapack/getrf.hpp"
#include "tlapack/blas/nrm2.hpp"
#include "../../eigen/Eigen/Core"

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <algorithm>


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
template<typename T>
std::vector<T> convertPythonListToVector(std::vector<T>& vec,PyObject* pyList) {

    if (!PyList_Check(pyList)) return vec;

    Py_ssize_t size = PyList_Size(pyList);
    for (Py_ssize_t i = 0; i < size; ++i) {
        PyObject* item = PyList_GetItem(pyList, i);
        vec.push_back(static_cast<T>(PyFloat_AsDouble(item)));
    }

    return vec;
}

//-------------------------------------------------------------------

//this function will convert H into an upper triangular R and b into Q^Tb. Then we can solve Rx = Q^Tb outside this function
template <typename matrix_t, typename vector_t>
void Hessenberg_qr(matrix_t H, vector_t b, int size)
{
    // data traits
    using idx_t = size_type<matrix_t>;
    using scalar_t = scalar_type<type_t<matrix_t>, type_t<vector_t> >;
    using range = pair<idx_t, idx_t>;

    auto m = nrows(H);
    auto n = ncols(H);
    float c = 0.0;
    float s = 0.0;
    float temp = 0.0;

    auto da_num = n < size ? n : size-1;
    for(int i = 0; i < da_num; i++) {
        c = H(i,i);
        s = -H(i+1,i);
        H(i,i) = sqrt(H(i,i)*H(i,i) + H(i+1,i)*H(i+1,i));
        c = c/H(i,i);
        s = s/H(i,i);
        H(i+1,i) = 0.0;
        for(int j = i+1; j < n; j++) {
            temp = c*H(i,j) - s*H(i+1,j);
            H(i+1,j) = s*H(i,j) + c*H(i+1,j);
            H(i,j) = temp;
        }
        temp = c*b[i] - s*b[i+1];
        b[i+1] = s*b[i] + c*b[i+1];
        b[i] = temp;
        
    }
    

}







//------------------------------------------------------------------------------
//this function performs step k of arnoldi iter where k > 0
template <TLAPACK_MATRIX matrixA_t, TLAPACK_MATRIX matrixH_t, TLAPACK_MATRIX matrixQ_t>
void arnoldi_iter(matrixA_t& A, matrixH_t& H, matrixQ_t& Q, int k)
{
    // data traits
    using idx_t = size_type<matrixA_t>;
    using scalar_t = scalar_type<type_t<matrixA_t>, type_t<matrixA_t> >;
    using range = pair<idx_t, idx_t>;
    // constants
    const idx_t n = nrows(Q);
    const idx_t m = ncols(H);

    // temporary storage
    std::vector<scalar_t> w(n);

    // one step of Arnoldi iteration
    // w = A * V[j]
        auto vec = slice(Q, range{0, m} ,k);
        gemv(Op::NoTrans, static_cast<scalar_t>(1.0), A, vec, static_cast<scalar_t>(0), w);

        // H[j,0:j+1] = V[0:n] * w
        for (idx_t i = 0; i < k+1; ++i)
            H(i, k) = dot(slice(Q, range{0, m} ,i), w);

        // w = w - V[0:n] * H[0:j+1,j]
        for (idx_t i = 0; i < k+1; ++i)
            axpy(-H(i, k), slice(Q, range{0, m} ,i), w);
            
        if(k == n-1) return;
        // H[k+1,k] = ||w||
        H(k+1, k) = nrm2(w);

        // Q[k+1] = w / H[k+1,k]
       
        rscl(H(k+1, k), w);
        for(int i = 0; i < m; i++) {
            Q(i,k+1) = w[i];
        }
        
    
}










//------------------------------------------------------------------------------

template<typename T, typename matrix_t>
int constructMatrix(int n, float cond, int space, bool geom, matrix_t& A, int p, float& true_cond) {
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
            pArgs = PyTuple_New(5);
            for (i = 0; i < 5; ++i) {
                switch(i) {
                    case 0:
                        pValue = PyLong_FromLong(n);
                        break;
                    case 1:
                        pValue = PyFloat_FromDouble(cond);
                        break;
                    case 2:
                        pValue = PyLong_FromLong(space);
                        break;
                    case 3:
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
                std::vector<T> b(n*n);
                std::vector<T> c(n*n + 1);
                for(int i = 0 ; i < n; i++) {
                    for(int j = 0; j < n; j++){
                        A(i,j) = static_cast<float>(static_cast<T>(PyFloat_AsDouble(PyList_GetItem(pValue, n*i + j))));
                    }
                }

                
                // tlapack::LegacyMatrix<T, int> LU(n, n, b.data(), n);
                // printMatrix(LU);
                true_cond = PyFloat_AsDouble(PyList_GetItem(pValue, n*n));
                std::cout << true_cond << std::endl;
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


std::ofstream myfile("e5m2_error_f_cond.csv");
template <typename T>
float run(size_t n, T scale, float cond, int p, int variant = 0) 
{   
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

    std::vector<float> Zero_(n * n);
    tlapack::LegacyMatrix<float, idx_t> Zero(n, n, Zero_.data(), n);


    float true_cond;
    constructMatrix<T>(n, cond, std::ceil(cond/static_cast<float>(5)) > n-1 ? n-1 : std::ceil(cond/static_cast<float>(5)) , true, FG, p, true_cond);
    
    std::vector<float> S_(n*n, 0.0);
    tlapack::LegacyMatrix<float, idx_t> S(n, n, S_.data(), n);
    for (size_t i = 0; i < n; i++){
        S(i, i) = 1.0;
    }
    std::vector<float> R_(n*n, 0.0);
    tlapack::LegacyMatrix<float, idx_t> R(n, n, R_.data(), n);
    for (size_t i = 0; i < n; i++){
        R(i, i) = 1.0;
    }

    float maxR, maxS;
    std::vector<float> AR(n), AS(n);

     int not_count = 0;
    while(false){
        for(int i = 0; i < n; i++){
            auto c1 = tlapack::rows(FG,range(i,i+1));
            AR[i] = 1/sqrt(tlapack::lange(tlapack::Norm::One, c1));
            auto c2 = tlapack::cols(FG, range(i,i+1));
            AS[i] = 1/sqrt(tlapack::lange(tlapack::Norm::One, c2));
            maxR = AR[i] > maxR ? AR[i] : maxR;
            maxS = AS[i] > maxS ? AS[i] : maxS;

        }
        for(int j = 0; j < n; j++){
            for(int k = 0; k < n; k++){
                FG(j,k) = FG(j,k)*(AR[j])*(AS[k]);
            }
        }
        for(int i = 0 ; i < n; i++){
            R(i,i) = R(i,i)*AR[i];
            S(i,i) = S(i,i)*AS[i];
        }
        //std::cout << maxR;
        not_count++;
        if(abs(maxR - 1) < 1 || abs(maxS - 1) < 1 || not_count > 100) break;
        }

        //next we need to scale by a parameter theta
        float maxA = tlapack::lange(tlapack::Norm::Max, FG);

        
        float normA = tlapack::lange(tlapack::Norm::Inf, FG);

    //first we'll get cond aafter preconditioning
    
    for (size_t j = 0; j < n; ++j){
        for (size_t i = 0; i < n; ++i){
            //A(i,j) = static_cast<real_t>(sqrt(float(scale)*0.125)*FG(i,j)/normA);
            A(i,j) = static_cast<real_t>(FG(i,j));
            Zero(i,j) = 0.0;
        }
     }

    //now generate all vectors
    std::vector<float> x_(n);
    tlapack::LegacyVector<float, idx_t> x(n, x_.data());

    std::vector<float> b_(n);
    tlapack::LegacyVector<float, idx_t> b(n, b_.data());

    std::vector<float> b1_(n);
    tlapack::LegacyVector<float, idx_t> b1(n, b1_.data());

    std::vector<float> b2_(n);
    tlapack::LegacyVector<float, idx_t> b2(n, b2_.data());

    std::vector<float> solved_r_(n);
    tlapack::LegacyVector<float, idx_t> solved_r(n, solved_r_.data());

    std::vector<float> bd_(n);
    tlapack::LegacyVector<float, idx_t> bd(n, bd_.data());

    std::vector<float> r_(n);
    tlapack::LegacyVector<float, idx_t> r(n, r_.data());

    std::vector<float> be_1_(n);
    tlapack::LegacyVector<float, idx_t> be_1(n, be_1_.data());

    for( int i = 0; i < n; i++) {
        b[i] = static_cast<float>(static_cast<real_t>(-0.5*(static_cast<float>(rand()))*float(scale)/static_cast<float>(RAND_MAX)));
        b[i] += static_cast<float>(static_cast<real_t>(static_cast<float>(rand())*float(scale)/static_cast<float>(RAND_MAX)));
        b1[i] = b[i];
        b2[i] = b[i];
        bd[i] = b[i];
        be_1[i] = (i == 0 ? 1.0 : 0.0);
    }

    //perform LU on A and FG
    std::vector<T> LU_(n * n);
    tlapack::LegacyMatrix<T, idx_t> LU(n, n, LU_.data(), n);

    std::vector<float> LU_copy_(n * n);
    tlapack::LegacyMatrix<float, idx_t> LU_copy(n, n, LU_copy_.data(), n);

    std::vector<float> LU_float_(n * n);
    tlapack::LegacyMatrix<float, idx_t> LU_float(n, n, LU_float_.data(), n);

    std::vector<float> LU_double_(n * n);
    tlapack::LegacyMatrix<float, idx_t> LU_double(n, n, LU_double_.data(), n);

    std::vector<float> LU_abs_(n * n);
    tlapack::LegacyMatrix<float, idx_t> LU_abs(n, n, LU_abs_.data(), n);

    std::vector<float> X_(n * n);
    tlapack::LegacyMatrix<float, idx_t> X(n, n, X_.data(), n);

    std::vector<float> X_abs_(n * n);
    tlapack::LegacyMatrix<float, idx_t> X_abs(n, n, X_abs_.data(), n);

    std::vector<float> H_(n*n);
    tlapack::LegacyMatrix<float, idx_t> H(n, n, H_.data(), n);

    std::vector<float> H_copy_(n*n);
    tlapack::LegacyMatrix<float, idx_t> H_copy(n, n, H_copy_.data(), n);

    std::vector<float> Q_(n*n);
    tlapack::LegacyMatrix<float, idx_t> Q(n, n, Q_.data(), n);

    //initialize first col of Q to be normalized b
    float normb = tlapack::nrm2(b);
    for(int i = 0; i < n; i++) {
        Q(i,0) = b[i]/normb;
    }



   
    // Matrix A is kept unchanged
    tlapack::lacpy(tlapack::GENERAL, A, LU);
    tlapack::lacpy(tlapack::GENERAL, FG, LU_float);
    tlapack::lacpy(tlapack::GENERAL, FG, LU_double);

    //declare arrays for piv
    std::vector<size_t> piv_lo(n);
    std::vector<size_t> piv_hi(n);

    int info, infotoo;
    if(variant == 0)    info = tlapack::getrf(LU, piv_lo, tlapack::GetrfOpts{GetrfVariant::Recursive});
    else info = tlapack::getrf(LU, piv_lo, tlapack::GetrfOpts{GetrfVariant::Level0});
    if (info != 0) {
        std::cerr << "Matrix could not be factorized :(" << std::endl;
        return -1;
    }
    if(variant == 0)    infotoo = tlapack::getrf(LU_float, piv_hi, tlapack::GetrfOpts{GetrfVariant::Recursive});
    else infotoo = tlapack::getrf(LU_float, piv_hi, tlapack::GetrfOpts{GetrfVariant::Level0});
    if (infotoo != 0) {
        std::cerr << "Matrix could not be factorized in fp32 :(" << std::endl;
        return -1;
    }
    int infotoo2 = tlapack::getrf(LU_double, piv_hi);
    if (infotoo2 != 0) {
        std::cerr << "Matrix could not be factorized in fp64 :(" << std::endl;
        return -1;
    }

    //compute sol in double

    for (idx_t i = 0; i < n;i++){
        if (piv_hi[i] != i) {
            auto tmp = bd[piv_hi[i]];
            bd[piv_hi[i]] = bd[i];
            bd[i] = tmp;
        }
    }

    tlapack::trsv(Uplo::Lower, NO_TRANS, Diag::Unit, LU_double, bd);
    tlapack::trsv(Uplo::Upper, NO_TRANS, Diag::NonUnit, LU_double, bd);

    for (idx_t i = 0; i < n;i++){
        if (piv_lo[i] != i) {
            auto tmp = b1[piv_lo[i]];
            b1[piv_lo[i]] = b1[i];
            b1[i] = tmp;
        }
    }
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            LU_copy(i,j) = static_cast<float>(LU(i,j));
        }
    }


    tlapack::trsv(Uplo::Lower, NO_TRANS, Diag::Unit, LU_copy, b1);
    tlapack::trsv(Uplo::Upper, NO_TRANS, Diag::NonUnit, LU_copy, b1);
    //before we begin proper IR, we need to make a copy of A called X, and in it, store inv(U)*inv(L)*A
    tlapack::lacpy(tlapack::GENERAL, FG, X);
    
   
    //permute X
    for (idx_t i = 0; i < n;i++){
        if (piv_lo[i] != i) {
            auto vect1 = tlapack::row(X, piv_lo[i]);
            auto vect2 = tlapack::row(X, i);
            tlapack::swap(vect1, vect2);
        }
    }
    tlapack::trsm(Side::Left, Uplo::Lower, NO_TRANS, Diag::Unit, 1.0,LU_copy, X);
    tlapack::trsm(Side::Left, Uplo::Upper, NO_TRANS, Diag::NonUnit, 1.0,LU_copy, X);
    
    //now we can begin the actual IR
    float res_norm = 1.0;
    float inner_res_norm = 1.0;

    int count = 0; //to keep track of number of IR iterations

    int num_iter = 0; //number of iters for GMRES
    float tol = std::pow(10,-6);
    for(int i = 0; i < n; i++) x[i] = b1[i];
    do {
        count = count + 1;
        for(int i = 0; i < n; i++) 
        {
        r[i] = b[i]; 
        }
        tlapack::gemv(NO_TRANS, -1.0, FG, x, 1.0, r);
        float m;
        res_norm = 0.0;
        for(int i = 0; i < n; i++) {
            m = m > abs(bd[i]) ? m : abs(bd[i]);
            res_norm = res_norm > abs(r[i]) ? res_norm : abs(r[i]);
        }

        res_norm = res_norm/(tlapack::lange(tlapack::INF_NORM, FG)*m);  
        myfile << count << "," << res_norm << std::endl;
        
        //copy r to solved_r
        for(int i = 0; i < n; i++) {
            solved_r[i] = r[i];
        }
        //condition solved_r by calls to trsv
         for (idx_t i = 0; i < n;i++){
        if (piv_lo[i] != i) {
            auto tmp = solved_r[piv_lo[i]];
            solved_r[piv_lo[i]] = solved_r[i];
            solved_r[i] = tmp;
        }
    }

        trsv(Uplo::Lower, NO_TRANS, Diag::Unit, LU_copy, solved_r);
        trsv(Uplo::Upper, NO_TRANS, Diag::NonUnit, LU_copy, solved_r);

        //now, using preconditioned r and A, perform GMRES
        //first get ||r|| for GMRES
       
      
        // lacpy(tlapack::GENERAL, Zero, Q);
        // lacpy(tlapack::GENERAL, Zero, H);
        normb = tlapack::nrm2(solved_r);
        //Now initialize first col of Q to be normalized b
        for(int i = 0; i < n; i++) {
            Q(i,0) = solved_r[i]/normb; 
        }
 
        while(num_iter < 5 && inner_res_norm > tol*normb) {
            //perform num_iterth step of arnoldi
            
            arnoldi_iter(X, H, Q, num_iter);
            num_iter = num_iter + 1; 
            for(int i = 0; i < n; i++) {
                for(int j = 0; j < n; j++) {
                    H_copy(i,j) = 0.0;
                }
            }
            tlapack::lacpy(tlapack::GENERAL, H, H_copy);

            // if(num_iter > 1) return 0.0;
            //solve ||Hx - b||
            
            for(int i = 0; i < m + n; i++) be_1[i] = (i == 0 ? normb : 0.0);
            if(num_iter != n) Hessenberg_qr(tlapack::slice(H_copy,range{0, num_iter+1}, range{0,num_iter}), tlapack::slice(be_1,range{0,  num_iter+1}), n);
            else  Hessenberg_qr(H_copy, be_1, n);
            
            auto da_tmp = tlapack::slice(be_1,range{0, num_iter});
            if(num_iter != n) tlapack::trsv(Uplo::Upper, tlapack::NO_TRANS,tlapack::Diag::NonUnit ,tlapack::slice(H_copy,range{0, num_iter}, range{0,num_iter}), da_tmp);
            else tlapack::trsv(Uplo::Upper, tlapack::NO_TRANS, tlapack::Diag::NonUnit, H_copy, da_tmp);
            //our solution vector is now obtained by multiplying by Q_n
            if(num_iter != n) tlapack::gemv(tlapack::NO_TRANS, 1.0, tlapack::slice(Q,range{0, n}, range{0,num_iter}), tlapack::slice(be_1, range{0, num_iter}), 0.0, solved_r);
            else tlapack::gemv(tlapack::NO_TRANS, 1.0, Q, be_1, 0.0, solved_r); 
        }
        //update r
        tlapack::axpy(1.0, solved_r, x);

        num_iter = 0;

        if(count > 10) break;



    } while(true);
    float FR = tlapack::nrm2(x);
    tlapack::axpy(-1.0, bd, x);
    return res_norm;




}




int main(int argc, char** argv) {
     typedef ml_dtypes::float8_e4m3fn float8e4m3fn;
    typedef ml_dtypes::float8_e5m2 float8e5m2;
    typedef ml_dtypes::float8_ieee_p<4> floate4m3;
    typedef ml_dtypes::float8_ieee_p<3> floate5m2;
    typedef ml_dtypes::block_float8_ieee<4> bfp;
    int m, n;

    std::cout << std::scientific << std::endl;
    
    // Default arguments
    n = atoi(argv[1]);
    float err1 = 0;
    float er3 = 0;
    float err2 = 0;


    if(atoi(argv[2]) == 0)
    er3 += run<floate4m3>(n, ml_dtypes::float8_internal::numeric_limits_float8_ieee_p<4>::max()/floate4m3{2.0}, static_cast<float>(atoi(argv[3])), 4, atoi(argv[4]));    
    else if(atoi(argv[2]) == 1)
    er3 += run<floate5m2>(n, ml_dtypes::float8_internal::numeric_limits_float8_ieee_p<3>::max(), static_cast<float>(atoi(argv[3])), 3, atoi(argv[4]));  
    else if(atoi(argv[2]) == 2)
    er3 +=   run<float>(n,1.0, static_cast<float>(atoi(argv[3])), 4, atoi(argv[4]));
    else if(atoi(argv[2]) == 3)
    er3 += run<bfp>(n,bfp(1.0), static_cast<float>(atoi(argv[3])), 4, atoi(argv[4]));
    else if(atoi(argv[2]) == 4)
    er3 += run<float8e4m3fn>(n, ml_dtypes::float8_internal::numeric_limits_float8_e4m3fn::max(), static_cast<float>(atoi(argv[3])), 4, atoi(argv[4]));
    else 
    er3 += run<int>(n,1.0, static_cast<float>(atoi(argv[3])), 6, atoi(argv[4]));
    
    
    std::cout << er3 << std::endl;
    return 0;
}