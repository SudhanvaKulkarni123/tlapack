#define TLAPACK_PREFERRED_MATRIX_LEGACY
#include <tlapack/plugins/legacyArray.hpp>
#include <tlapack/plugins/stdvector.hpp>
#include <tlapack/blas/dotu.hpp>
#include <tlapack/blas/gemm.hpp>

template<typename matrix_t>
void orthog(int m, int n, matrix_t& A) {

}