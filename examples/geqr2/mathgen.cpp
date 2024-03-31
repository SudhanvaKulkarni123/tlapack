#define TLAPACK_PREFERRED_MATRIX_LEGACY
#include <tlapack/plugins/legacyArray.hpp>
#include <tlapack/plugins/stdvector.hpp>
#include <tlapack/blas/dotu.hpp>
#include <tlapack/blas/gemm.hpp>

template<typename matrix_t>
void orthog(int m, int n, matrix_t& Q) {
    std::random_device generator;
    std::normal_distribution<double> distribution(0.0,1.0);
    std::vector<float> A_u(n*m, 0.0);

    std::vector<float> tau(n);
    std::vector<float> tau_buffer(n);

    tlapack::geqr2(R1, tau_buffer);

    tlapack::Create<matrix_t> new_matrix;
    matrix_t A = new_matrix(A_u,m,n);
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            A(i,j) = distribution(generator);
        }
    }


}