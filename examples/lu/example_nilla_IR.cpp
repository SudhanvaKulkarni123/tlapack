#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define TLAPACK_PREFERRED_MATRIX_LEGACY
#include <tlapack/plugins/legacyArray.hpp>
#include <tlapack/plugins/stdvector.hpp>
#include <tlapack/plugins/float8_iee_p.hpp>

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

template<typename T, typename matrix_t>
int constructMatrix(int n, float cond, int space, bool geom, matrix_t& A, int p) {
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
                std::cout << PyFloat_AsDouble(PyList_GetItem(pValue, 6)) << std::endl;
                for(int i = 0 ; i < n; i++) {
                    for(int j = 0; j < n; j++){
                        A(i,j) = static_cast<float>(static_cast<T>(PyFloat_AsDouble(PyList_GetItem(pValue, n*i + j))));
                    }
                }
                
                // tlapack::LegacyMatrix<T, int> LU(n, n, b.data(), n);
                // printMatrix(LU);
                std::cout << PyFloat_AsDouble(PyList_GetItem(pValue, n*n)) << std::endl;
                std::cout << std::pow(2.0,-23) << std::endl;
                std::cout << PyFloat_AsDouble(PyList_GetItem(pValue, n*n)) * std::pow(2.0,-22) << std::endl;
                
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
std::ofstream other_file("debug.csv");
template <class T>
double run(size_t n, T scale, float cond, int p) {

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
            FG(i,j) = (float(scale)*static_cast<float>(rand()))/static_cast<float>(RAND_MAX);
            // A(i,j) = static_cast<T>(FG(i,j));
        }
    }
    constructMatrix<T>(n, cond, std::ceil(cond/static_cast<float>(5)) > n-1 ? n-1 : std::ceil(cond/static_cast<float>(5)) , true, FG, p);
    other_file << cond << std::endl;

    for (idx_t i = 0; i <n; ++i) {
        other_file << std::endl;
        for (idx_t j = 0; j < n; ++j)
            other_file << FG(i, j) << " ";
    }
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
        }
     }

    //now generate n dimensional vector b to solve Ax = b
    std::vector<float> b_(n);
    tlapack::LegacyVector<float, idx_t> b(n, b_.data());

    std::vector<float> b1_(n);
    tlapack::LegacyVector<float, idx_t> b1(n, b1_.data());

    std::vector<float> b2_(n);
    tlapack::LegacyVector<float, idx_t> b2(n, b2_.data());

    std::vector<float> bd_(n);
    tlapack::LegacyVector<float, idx_t> bd(n, bd_.data());

    std::vector<float> r_(n);
    tlapack::LegacyVector<float, idx_t> r(n, r_.data());

    for( int i = 0; i < n; i++) {
        b[i] = static_cast<float>(static_cast<real_t>(-0.5*(static_cast<float>(rand()))*float(scale)/static_cast<float>(RAND_MAX)));
        b[i] += static_cast<float>(static_cast<real_t>(static_cast<float>(rand())*float(scale)/static_cast<float>(RAND_MAX)));
        b1[i] = b[i];
        b2[i] = b[i];
        bd[i] = b[i];
    }

    //perform LU on A and FG
    std::vector<T> LU_(n * n);
    tlapack::LegacyMatrix<T, idx_t> LU(n, n, LU_.data(), n);

    std::vector<float> LU_float_(n * n);
    tlapack::LegacyMatrix<float, idx_t> LU_float(n, n, LU_float_.data(), n);

    std::vector<double> LU_double_(n * n);
    tlapack::LegacyMatrix<double, idx_t> LU_double(n, n, LU_double_.data(), n);

    std::vector<float> LU_abs_(n * n);
    tlapack::LegacyMatrix<float, idx_t> LU_abs(n, n, LU_abs_.data(), n);

    std::vector<float> X_(n * n);
    tlapack::LegacyMatrix<float, idx_t> X(n, n, X_.data(), n);

    std::vector<float> X_abs_(n * n);
    tlapack::LegacyMatrix<float, idx_t> X_abs(n, n, X_abs_.data(), n);

   
    // Matrix A is kept unchanged
    tlapack::lacpy(tlapack::GENERAL, A, LU);
    tlapack::lacpy(tlapack::GENERAL, FG, LU_float);
    tlapack::lacpy(tlapack::GENERAL, FG, LU_double);

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
    int infotoo2 = tlapack::getrf(LU_double, piv_hi);
    if (infotoo2 != 0) {
        std::cerr << "Matrix could not be factorized in fp64 :(" << std::endl;
        return -1;
    }
    other_file << "-----------------------------------------------------------------------------------------------------------" << std::endl;
     for (idx_t i = 0; i <n; ++i) {
        other_file << std::endl;
        for (idx_t j = 0; j < n; ++j)
            other_file << LU(i, j) << " ";
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


    //create copy to stor low prec LU in high prec
    std::vector<float> LU_copy_(n * n);
    tlapack::LegacyMatrix<float, idx_t> LU_copy(n, n, LU_copy_.data(), n);

     for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            LU_copy(i,j) = static_cast<float>(LU(i,j));
            LU_abs(i,j) = abs(static_cast<float>(LU(i,j)));
        }
    }

    for (idx_t i = 0; i < n;i++){
        if (piv_lo[i] != i) {
            auto tmp = b1[piv_lo[i]];
            b1[piv_lo[i]] = b1[i];
            b1[i] = tmp;
        }
    }

    for(int i = 0; i < n; i++) {
        X_abs(i,i) = 1.0;
        X(i,i) = 1.0;
    }

    tlapack::trmm(tlapack::Side::Left, tlapack::Uplo::Upper,
                  tlapack::Op::NoTrans, tlapack::Diag::NonUnit, 1.0, LU_abs,
                  X_abs);

    tlapack::trmm(tlapack::Side::Left, tlapack::Uplo::Lower,
                  tlapack::Op::NoTrans, tlapack::Diag::Unit, 1.0, LU_abs, X_abs);

   
    tlapack::trmm(tlapack::Side::Left, tlapack::Uplo::Upper,
                  tlapack::Op::NoTrans, tlapack::Diag::NonUnit, 1.0, LU_copy,
                  X);

    tlapack::trmm(tlapack::Side::Left, tlapack::Uplo::Lower,
                  tlapack::Op::NoTrans, tlapack::Diag::Unit, 1.0, LU_copy, X);
     
     for (idx_t i = n; i-- > 0;) {
        if (piv_lo[i] != i) {
            auto vect1 = tlapack::row(X, piv_lo[i]);
            auto vect2 = tlapack::row(X, i);
            tlapack::swap(vect1, vect2);
            auto vect3 = tlapack::row(X_abs, piv_lo[i]);
            auto vect4 = tlapack::row(X_abs, i);
            tlapack::swap(vect3, vect4);
        }
    }
    float maxim = 0.0;
    float maxu = 0.0;
    for(int i = 0; i < n; i++) {
        maxu = maxu > LU_abs(i,i) ? maxim : LU_abs(i,i);
    }
    for(int i = 0; i  < n; i++) {
        for(int j = 0; j < n; j++) {
            auto tmp = (abs(float(A(i,j)) - X(i,j)))/X_abs(i,j);
            maxim = maxim > tmp ? maxim : tmp;
        }
    }
    //return maxim;

    tlapack::trsv(Uplo::Lower, NO_TRANS, Diag::Unit, LU_copy, b1);
    tlapack::trsv(Uplo::Upper, NO_TRANS, Diag::NonUnit, LU_copy, b1);

    std::cout << "b1 before we begin IR :" << b1[0] << "," << b1[1] << "," << b1[2] << "," << b1[3] << "," << b1[4] << "," << std::endl;

    //first get sol in 32
    for (idx_t i = 0; i < n;i++){
        if (piv_hi[i] != i) {
            auto tmp = b2[piv_hi[i]];
            b2[piv_hi[i]] = b2[i];
            b2[i] = tmp;
        }
    }

    tlapack::trsv(Uplo::Lower, NO_TRANS, Diag::Unit, LU_float, b2);
    tlapack::trsv(Uplo::Upper, NO_TRANS, Diag::NonUnit, LU_float, b2);
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
        float m;
        res_norm= 0.0;
        for(int i = 0; i < n; i++) {
            m = m > abs(bd[i]) ? m : abs(bd[i]);
            res_norm = res_norm > abs(r[i]) ? res_norm : abs(r[i]);
        }
        res_norm = res_norm/(tlapack::lange(tlapack::INF_NORM, FG)*m);   
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
        //plot err vs iteration
        myfile << count << "," << res_norm << std::endl;
        if(count > 30) break;
        
    } while(true);


    

    std::cout << "result :" << b1[0] << "," << b1[1] << "," << b1[2] << "," << b1[3] << "," << b1[4] << "," << std::endl;
    std::cout << "other result :" << b2[0] << "," << b2[1] << "," << b2[2] << "," << b2[3] << "," << b2[4] << "," << std::endl;
    float max = 0.0;
    for(int i = 0; i < n; i++) {
        max = max > abs(b2[i]) ? max : abs(b2[i]);
    }
    tlapack::gemv(tlapack::NO_TRANS, 1.0, FG, b1, -1.0, b);
    //now b has residual
    std::cout << tlapack::nrm2(b) << std::endl;
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
    std::vector<floate4m3> A_(5 * 5);
    tlapack::LegacyMatrix<floate4m3, int> A(5, 5, A_.data(), 5);

    std::cout << std::scientific << std::endl;
    
    


    // Default arguments
    n = atoi(argv[1]);
    double err1 = 0;
    double er3 = 0;
    double err2 = 0;

    

    if(atoi(argv[2]) == 0)
    er3 += run<floate4m3>(n, ml_dtypes::float8_internal::numeric_limits_float8_ieee_p<4>::max()/floate4m3{2.0}, static_cast<float>(atoi(argv[3])), 4);    
    else if(atoi(argv[2]) == 1)
    er3 += run<floate5m2>(n, ml_dtypes::float8_internal::numeric_limits_float8_ieee_p<3>::max(), static_cast<float>(atoi(argv[3])), 3);  
    else if(atoi(argv[2]) == 2)
    er3 +=   run<float>(n,1.0, static_cast<float>(atoi(argv[3])), 4);
    else if(atoi(argv[2]) == 3)
    er3 += run<bfp>(n,bfp(1000.0), static_cast<float>(atoi(argv[3])), 4);
    else if(atoi(argv[2]) == 4)
    er3 += run<float8e4m3fn>(n, ml_dtypes::float8_internal::numeric_limits_float8_e4m3fn::max(), static_cast<float>(atoi(argv[3])), 4);    
    else 
    er3 += run<int>(n,1.0, static_cast<int>(atoi(argv[3])), 6);
    
    

    std::cout << "err bound is " << er3 << std::endl;




    return 0;
}