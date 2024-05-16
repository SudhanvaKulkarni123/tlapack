#define PY_SSIZE_T_CLEAN
#include <Python.h>
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

template <typename matrixA_t, typename matrixB_t>
bool MatricesEqual(const matrixA_t& A, const matrixB_t& B)
{
    using idx_t = tlapack::size_type<matrixA_t>;
    const idx_t m = tlapack::nrows(A);
    const idx_t n = tlapack::ncols(A);
    bool to_ret = true;

    for (idx_t i = 0; i < m; ++i) {
        for (idx_t j = 0; j < n; ++j)
            to_ret = to_ret & (float(A(i,j)) == float(B(i,j)));
    }
    return to_ret;
}

template <typename vec_t>
void printVector(const vec_t v, int n)
{
    for(int i = 0; i < n; i++) {
        std::cout << v[i] << "," << std::endl;
    }
}


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
                
                
                
             //get computed condition number
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


std::ofstream myfile("e5m2_error_f_cond.csv");
template <typename real_t>
double run(size_t m, size_t n, real_t scale, float cond, int name, bool arithmetic, int p)
{
    using std::size_t;
    using matrix_t = tlapack::LegacyMatrix<real_t>;
    using idx_t = int;
    using range = pair<idx_t, idx_t>;
    // Functors for creating new matrices
    tlapack::Create<matrix_t> new_matrix;
 
    // Turn it off if m or n are large
    bool verbose = false;

    // Arrays
    std::vector<real_t> tau(m);
    std::vector<float> tau_f(m);
    std::vector<float> scal(n);

    //declare all required matrices
    std::vector<float> FG_;
    auto FG = new_matrix(FG_, m, n);
    std::vector<double> FGd_;
    auto FGd = new_matrix(FGd_, m, n);
    std::vector<float> FGi_;
    auto FGi = new_matrix(FGi_, n, m);
    std::vector<float> FGTFG_;
    auto FGTFG = new_matrix(FGTFG_, n, n);
    std::vector<double> A_abs_;
    auto A_abs = new_matrix(A_abs_, m, n);
    std::vector<float> A_aug_;
    auto A_aug = new_matrix(A_aug_, m+n, m+n);
    std::vector<real_t> A_;
    auto A = new_matrix(A_, m, n);
    std::vector<real_t> R_;
    auto R = new_matrix(R_, m, n);
    std::vector<float> R_copy_;
    auto R_copy = new_matrix(R_copy_, m, n);
    std::vector<real_t> Q_;
    auto Q = new_matrix(Q_, m, m);
    std::vector<float> Q_copy_;
    auto Q_copy = new_matrix(Q_copy_, m, m);
    std::vector<float> Rf_;
    auto Rf = new_matrix(Rf_, m, n);
    std::vector<float> Qf_;
    auto Qf = new_matrix(Qf_, m, m);
    std::vector<float> Q1_;
    auto Q1 = new_matrix(Q1_, m, n);
    std::vector<float> Q2_;
    auto Q2 = new_matrix(Q2_, m, m-n);
    std::vector<double> Q1f_;
    auto Q1f = new_matrix(Q1f_, m, n);
    std::vector<double> Q2f_;
    auto Q2f = new_matrix(Q2f_, m, m-n);

    

    double cond_ref;
    constructMatrix<real_t>(m, n, cond, std::ceil(cond/static_cast<float>(5)) > n-1 ? n-1 : std::ceil(cond/static_cast<float>(5)) , true, FG, p, cond_ref, FGi, FGTFG);
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            A(i,j) = real_t(FG(i,j));
            FGd(i,j) = FG(i,j);
            FGi(j,i) = abs(FGi(j,i));
            if(i < n) FGTFG(i,j) = abs(FGTFG(i,j));
            A_abs(i,j) = abs(FG(i,j));
        }
    }

    auto normA = tlapack::lange(INF_NORM, FGd);

     //now generate n dimensional vector b to solve min ||Ax - b||_2
    std::vector<float> b_(m);
    tlapack::LegacyVector<float, idx_t> b(m, b_.data());

    std::vector<float> abs_b_(m);
    tlapack::LegacyVector<float, idx_t> abs_b(m, abs_b_.data());
    //residual
    std::vector<float> r_(m);
    tlapack::LegacyVector<float, idx_t> r(m, r_.data());
    //residual copy
    std::vector<float> r_copy_(m);
    tlapack::LegacyVector<float, idx_t> r_copy(m, r_copy_.data());
    //vector with absolute values of r
    std::vector<float> abs_r_(m);
    tlapack::LegacyVector<float, idx_t> abs_r(m, abs_r_.data());
    //buffer vectors
    std::vector<float> s_(m);
    tlapack::LegacyVector<float, idx_t> s(m, s_.data());

    std::vector<float> t_(n);
    tlapack::LegacyVector<float, idx_t> t(n, t_.data());

    std::vector<float> u_(m);
    tlapack::LegacyVector<float, idx_t> u(m, u_.data());

    std::vector<float> u_buf_(m);
    tlapack::LegacyVector<float, idx_t> u_buf(m, u_buf_.data());

    std::vector<float> v_(n);
    tlapack::LegacyVector<float, idx_t> v(n, v_.data());

    std::vector<float> c_(n);
    tlapack::LegacyVector<float, idx_t> c(n, c_.data());

    std::vector<float> d_(m - n);
    tlapack::LegacyVector<float, idx_t> d(m - n, d_.data());

    std::vector<float> e_(n);
    tlapack::LegacyVector<float, idx_t> e(n, e_.data());

    //soln vector
    std::vector<float> x_(n);
    tlapack::LegacyVector<float, idx_t> x(n, x_.data());
    //vector containing absolute values of x
    std::vector<float> abs_x_(n);
    tlapack::LegacyVector<float, idx_t> abs_x(n, abs_x_.data());

    //vector for backward error
    std::vector<double> err_vec_(n);
    tlapack::LegacyVector<double, idx_t> err_vec(n, err_vec_.data());

    std::vector<double> r_and_x_(m+n);
    tlapack::LegacyVector<double, idx_t> r_and_x(m+n, r_and_x_.data());



    for( int i = 0; i < m; i++) {
        b[i] = (static_cast<float>(rand()))/static_cast<float>(RAND_MAX);
        s[i] = b[i];
        r[i] = b[i];
        r_copy[i] = b[i];   
        abs_b[i] = abs(b[i]); 
        }
   

    for(int i = 0; i < n; i++) {
        t[i] = 0;
    }

    for(int i = 0; i < m+n; i++) {
        for(int j = 0; j < m+n; j++) {
            if(i < m & j< m) A_aug(i,j) = (i == j) ? 1.0 : 0.0;
            else if( i < m & j >= m) A_aug(i,j) = FG(i,j - m);
            else if( i >= m & j < m) A_aug(i,j) = FG(j, i - m);
            else A_aug(i,j) = 0.0;
        }
    }

    //okay now we may begin, first take QR of FG
    tlapack::lacpy(tlapack::GENERAL, A, Q);

   
    tlapack::geqr2(Q, tau, scal);

 

    tlapack::lacpy(tlapack::GENERAL, FGd, Qf);
    tlapack::geqr2(Qf, tau_f, scal);

    //save R

    for(int i = 0; i < m; i++) {
        for(int j =0; j < n; j++) {
            R(i,j) = Q(i,j);
            Rf(i,j) = Qf(i,j);
        }
    }

    

   

    for(int i = 0; i < m; i++){
            for(int j = 0; j < (i < n ? i : n); j++){
                R(i,j) = static_cast<real_t>(0.0);
                Rf(i,j) = 0;
            }
        }

    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            R_copy(i,j) = static_cast<float>(R(i,j));
        }
    }

    //time taken for ungr2

    tlapack::ung2r(Q, tau);

    tlapack::ung2r(Qf, tau_f);
    

    //now we need to spend some time making the augmented system

    int cnt = 0;
    auto r_norm = 0.0;
    auto dr_norm = 0.0;
    auto old_dr_norm = 0.0;

    //now split Q
    for(int i =0; i < m; i++) {
        for(int j = 0; j< n; j++) {
            Q1(i,j) = static_cast<float>(Q(i,j));
            Q_copy(i,j) = static_cast<float>(Q(i,j));
            Q1f(i,j) = Qf(i,j);
        }
    }

    for(int i = 0; i < m; i++) {
        for(int j  =n; j < m; j++) {
            Q2(i,j-n) = static_cast<float>(Q(i,j));
            Q_copy(i,j) = static_cast<float>(Q(i,j));
            Q2f(i, j-n) = Qf(i,j);
        }
    }

    auto guy = tlapack::slice(R_copy,range{0,n}, range{0,n});
    auto guyf = tlapack::slice(Rf,range{0,n}, range{0,n});
    std::vector<float> other_guy_(n*n);
    auto other_guy = new_matrix(other_guy_, n, n);
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            other_guy(i,j) = guy(j,i);
        }
    }
    std::vector<float> other_guy_f_(n*n);
    auto other_guy_f = new_matrix(other_guy_f_, n, n);
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            other_guy_f(i,j) = guyf(j,i);
        }
    }

    //first solve the problem in double precision to get "true solution"
    for(int i = 0; i < m; i++) s[i] = b[i];
    tlapack::gemv(tlapack::TRANSPOSE, 1.0, Q1f, s, 0.0, c);          //c = Q1^Ts        -- mn flops
    tlapack::gemv(tlapack::TRANSPOSE, 1.0, Q2f, s, 0.0, d);          //d= Q2^Ts         -- m(m - n) flops
    tlapack::trsv(Uplo::Lower, tlapack::NO_TRANS, Diag::NonUnit, other_guy_f, t);       //e is now stored in t      -- n*(n - 1) flops
    for(int i = 0; i < n; i++) {
         v[i] = -t[i];      //now v = -e
         e[i] = t[i];
         
    }
    tlapack::axpy(1.0, c, v);   //now v = inv(R)*(c-e)          -- n flops
    tlapack::trsv(Uplo::Upper, tlapack::NO_TRANS, Diag::NonUnit, guyf , v);         //-- n(n + 1) flops 
    for (int i = 0; i < m; i++) {
        if(i < n) u_buf[i] = e[i];
        else u_buf[i] = d[i - n];
    }
    tlapack::gemv(tlapack::NO_TRANS,1.0,Qf,u_buf,0.0,u);        //--- 2m^2 flops
    



    std::vector<double> true_sol(m+n);
    std::vector<double> true_x(n);
    double true_x_norm = 0.0;
    for(int i = 0 ; i < m+n; i++) {
        true_sol[i] = (i < m ? u[i] : v[i - m]); 
        if(i >= m) {
            true_x[i - m] = v[i - m];
            true_x_norm = true_x[i - m]*true_x[i - m];
        }
    }
    //now that we have true solution, we can compute the norm of the true solution and condition number
    

    for(int i = 0; i < m; i++) {
        abs_r[i] = abs(r[i]);
        if(i < n) abs_x[i] = abs(true_x[i]);
    }
    double maximim = 0.0;
    for(int i = 0; i < n; i++) {
        maximim = (maximim > abs_x[i]) ? maximim : abs_x[i];
    }
    //perform |A||x| + |b|
    tlapack::gemv(tlapack::NO_TRANS, 1.0, A_abs, abs_x, 1.0, abs_b);
    //perform |pinv(A)|*|b|
    tlapack::gemv(tlapack::NO_TRANS, 1.0, FGi, abs_b, 0.0, e);
    //perform |A|^T*abs_r
    tlapack::gemv(tlapack::TRANSPOSE, 1.0, A_abs, abs_r, 0.0, v);
    //perform FGTFG*abs_b
    tlapack::gemv(tlapack::NO_TRANS, 1.0, FGTFG, v, 0.0, abs_x);
    // |pinv(A)|*|b| + FGTFG*abs_b
    double nrm1 = 0.0;
    for(int i = 0; i < n; i++) {
        nrm1 = nrm1 > abs_x[i] ? nrm1 : abs_x[i];
    }
    double nrm2 = 0.0;
    for(int i = 0; i < n; i++) {
        nrm2 = nrm2 > e[i] ? nrm2 : e[i];
    }
    

    auto condition_num = (nrm1 + nrm2)/maximim; //now add this to the csv


   



    //now compute time for initial sol proceure
 
    for(int i = 0; i < m; i++) s[i] = b[i];
    
    tlapack::gemv(tlapack::TRANSPOSE, 1.0, Q1, s, 0.0, c);          //c = Q1^Ts
    tlapack::gemv(tlapack::TRANSPOSE, 1.0, Q2, s, 0.0, d);          //d= Q2^Ts
    
    for(int i = 0; i < n; i++) t[i] = 0;
    tlapack::trsv(Uplo::Lower, tlapack::NO_TRANS, Diag::NonUnit, other_guy, t);       //e is now stored in t
    for(int i = 0; i < n; i++) {
         v[i] = -t[i];      //now v = -e
         e[i] = t[i];
         
    }
    
    tlapack::axpy(1.0, c, v);   //now v = inv(R)*(c-e)
    tlapack::trsv(Uplo::Upper, tlapack::NO_TRANS, Diag::NonUnit, guy , v);
    
    for (int i = 0; i < m; i++) {
        if(i < n) u_buf[i] = e[i];
        else u_buf[i] = d[i - n];
    }
    tlapack::gemv(tlapack::NO_TRANS,1.0,Q_copy,u_buf,0.0,u);
    for(int i = 0; i < m; i++) {
        u[i] = u[i];
        if(i < n) v[i] = v[i];
    }
    

   

    //initial soln is u,v
    for(int i = 0; i < m; i++) {
        r[i] = u[i];
        r_copy[i] = u[i];
    }
    std::cout << std::endl;

    
    for(int i = 0; i < n; i++) {
        x[i] = v[i];
    }

    std::cout << std::endl;
    
    r_norm = 0.0;
    for(int i = 0; i < m; i++) {
        r_norm += u[i]*u[i];
    }
    auto x_norm = 0.0;
    auto dx_norm = 0.0;
    for(int i = 0; i < n; i++) {
        x_norm += v[i]*v[i];
    }

   auto s_norm = 0.0;
   auto maxS = 0;
    
    for(int i =0; i < m; i++) {
        for(int j = 0; j< n; j++) {
            Q1(i,j) = static_cast<float>(Q(i,j));
            Q_copy(i,j) = static_cast<float>(Q(i,j));
        }
    }

    for(int i = 0; i < m; i++) {
        for(int j  =n; j < m; j++) {
            Q2(i,j-n) = static_cast<float>(Q(i,j));
            Q_copy(i,j) = static_cast<float>(Q(i,j));
        }
    }
    auto r_and_x_norm = 0.0;
    double back_error = 0.0;


    myfile << cnt << "," << condition_num << std::endl;
    double max_x = 0.0;
    do {
        //time for one iter

        for(int i = 0; i < m; i++) {
            s[i] = r[i];
        }
        tlapack::axpy(-1.0, b, s);     
        
        // s = r - b
        tlapack::gemv(tlapack::NO_TRANS, -1.0, FG, x, -1.0, s);   //compute residual --> b - Ax - r
       
        // b - Ax - r
        tlapack::gemv(tlapack::TRANSPOSE, -1.0, FG, r, 0.0, t); //compute -A^Tr --> t = -A^Tr

  //c = Q1^Ts
        tlapack::gemv(tlapack::TRANSPOSE, 1.0, Q1, s, 0.0, c);          //c = Q1^Ts
        tlapack::gemv(tlapack::TRANSPOSE, 1.0, Q2, s, 0.0, d); 
   
        tlapack::trsv(Uplo::Lower, tlapack::NO_TRANS, Diag::NonUnit, other_guy, t);       //e is now stored in t --> e = inv(R^T)*t
       
        for(int i = 0; i < n; i++) {
            v[i] = c[i]-t[i];      //now v = c-e
            e[i] = t[i];
            
        }
        
        tlapack::trsv(Uplo::Upper, tlapack::NO_TRANS,Diag::NonUnit, guy, v);    //v = inv(R)*(c - e)
        for (int i = 0; i < m; i++) {
            if(i < n) u_buf[i] = e[i];
            else u_buf[i] = d[i - n];
        }

        tlapack::gemv(tlapack::NO_TRANS, 1.0, Q_copy, u_buf, 0.0, u);
        for(int i = 0; i < m; i++) {
            u[i] = u[i];
            if(i < n) v[i] = v[i];
        }
        //update soln vector
        tlapack::axpy(1.0, u, r);
        tlapack::axpy(1.0, v, x);
        //end time


   
        for(int i = 0; i < n; i++) err_vec[i] = (x[i]);
        tlapack::axpy(-1.0, true_x, err_vec);
        back_error = 0.0;
        for(int i = 0; i < n; i++) back_error = (back_error > abs(err_vec[i]) ? back_error : abs(err_vec[i]));
        back_error = back_error/(maximim);
        cnt++;
        //now at the end of the iter, compute abs(pinv(A))*(abs(b)  + abs(A)*abs(x)) + abs(inv(A^TA)*abs(A^T)*abs(r))

        myfile << cnt << "," << back_error << std::endl;
        if(cnt > 5) break;
        


    } while (true);
   


   
   std::cout << std::endl;
   for(int i = 0; i < m; i++) {
    s_norm += abs(s[i]*s[i]);
   }
   s_norm = sqrt(s_norm);

   
    





    
    
    //now post-process
    std::cout << "final residual error: " << s_norm  << std::endl;
    return condition_num;



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

    srand(atoi(argv[3]));  // Init random seed

    std::cout.precision(10);
    std::cout << std::scientific << std::showpos;

   
    if(atoi(argv[5]) == 0)
    er3 += run<floate4m3>(m, n, ml_dtypes::float8_internal::numeric_limits_float8_ieee_p<4>::max()/floate4m3{2.0}, static_cast<float>(atoi(argv[3])), 0, atoi(argv[4]) == 1, 4);    
    else if(atoi(argv[5]) == 1)
    er3 += run<floate5m2>(m, n, ml_dtypes::float8_internal::numeric_limits_float8_ieee_p<3>::max(), static_cast<float>(atoi(argv[3])), 0, atoi(argv[4]) == 1, 3);  
    else if(atoi(argv[5]) == 2)
    er3 +=   run<float>(m,n,1.0, static_cast<float>(atoi(argv[3])), 0, atoi(argv[4]) == 1, 4);
    else if(atoi(argv[5]) == 3)
    er3 += run<bfp>(m,n,bfp(1000.0), static_cast<float>(atoi(argv[3])), 0, atoi(argv[4]) == 1, 4);
    else if(atoi(argv[5]) == 4)
    er3 += run<float8e4m3fn>(m, n, ml_dtypes::float8_internal::numeric_limits_float8_e4m3fn::max(), static_cast<float>(atoi(argv[3])), 0, atoi(argv[4]) == 1, 4);    
    else 
    er3 += run<int>(m,n,1.0, static_cast<int>(atoi(argv[3])), 0, atoi(argv[4]) == 1, 4);
    
    
    return 0;

}