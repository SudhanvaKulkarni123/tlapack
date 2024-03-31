#ifndef TLAPACK_ARNOLDI_HH
#define TLAPACK_ARNOLDI_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/gemv.hpp"
#include "tlapack/blas/dot.hpp"
#include "tlapack/blas/axpy.hpp"
#include "tlapack/blas/nrm2.hpp"
/**
* @param[in] A $op(A)$ is an m-by-k matrix.
**/

namespace tlapack {
    template <TLAPACK_MATRIX matrixA_t, TLAPACK_MATRIX matrixH_t,
     TLAPACK_VECTOR vectorX_t>
    void arnoldi_iter(const matrixA_t& A, matrixH_t& H, vectorX_t& v) {
        for(int i = 1; i < A.rows(); i++){
            gemv(tlapack::NO_TRANS, 1.0, A, q, v);
            for(int j = 0; j < i; j++) {
                H(j,i-1) = dot(q,q);



            }
        }


    }
     
}

#endif