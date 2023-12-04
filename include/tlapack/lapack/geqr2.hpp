/// @file geqr2.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// @note Adapted from @see
/// https://github.com/langou/latl/blob/master/include/geqr2.h
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_GEQR2_HH
#define TLAPACK_GEQR2_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/larf.hpp"
#include "tlapack/lapack/larfg.hpp"

namespace tlapack {

/** Worspace query of geqr2()
 *
 * @param[in] A m-by-n matrix.
 *
 * @param tau Not referenced.
 *
 * @return WorkInfo The amount workspace required.
 *
 * @ingroup workspace_query
 */
template <class T, TLAPACK_SMATRIX matrix_t, TLAPACK_VECTOR vector_t>
constexpr WorkInfo geqr2_worksize(const matrix_t& A, const vector_t& tau)
{
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    // constants
    const idx_t n = ncols(A);

    if (n > 1) {
        auto&& C = cols(A, range{1, n});
        return larf_worksize<T>(LEFT_SIDE, FORWARD, COLUMNWISE_STORAGE,
                                col(A, 0), tau[0], C);
    }

    return WorkInfo(0);
}

/** @copybrief geqr2()
 * Workspace is provided as an argument.
 * @copydetails geqr2()
 *
 * @param work Workspace. Use the workspace query to determine the size needed.
 *
 * @ingroup computational
 */
template <TLAPACK_SMATRIX matrix_t,
          TLAPACK_VECTOR vector_t,
          TLAPACK_WORKSPACE work_t>
int geqr2_work(matrix_t& A, vector_t& tau, work_t& work)
{   
    using namespace std;
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t k = std::min<idx_t>(m, n - 1);

    // check arguments
    tlapack_check_false((idx_t)size(tau) < min(m, n));

    // quick return
    if (n <= 0 || m <= 0) return 0;
    #ifdef TESTSCALING
    // int count = 0;          //parameter to set for how often we scale the matrix
    // vector<float> SM_(n, 1.0);
    // auto ScalMat = new_matrix(SM_, n, n);   //diagonal matrix that we'll multiply to the right of our matrix so that we end up with QRD instead of QR
    // for(int i = 0 ; i < n; i++){
    //     ScalMat(i,i) = 1;
    // }
    #endif
    for (idx_t i = 0; i < k; ++i) {
        // Define v := A[i:m,i]
        auto v = slice(A, range{i, m}, i);

        // Generate the (i+1)-th elementary Householder reflection on v
        larfg(FORWARD, COLUMNWISE_STORAGE, v, tau[i]);

        // Define C := A[i:m,i+1:n]
        auto C = slice(A, range{i, m}, range{i + 1, n});
        #ifdef TESTSCALING
        // diagscal()

        #endif
        // C := ( I - conj(tau_i) v v^H ) C
        larf_work(LEFT_SIDE, FORWARD, COLUMNWISE_STORAGE, v, conj(tau[i]), C,
                  work);
    }
    if (n - 1 < m) {
        // Define v := A[n-1:m,n-1]
        auto v = slice(A, range{n - 1, m}, n - 1);
        // Generate the n-th elementary Householder reflection on v
        larfg(FORWARD, COLUMNWISE_STORAGE, v, tau[n - 1]);
    }

    return 0;
}

/** Computes a QR factorization of a matrix A.
 *
 * The matrix Q is represented as a product of elementary reflectors
 * \[
 *          Q = H_1 H_2 ... H_k,
 * \]
 * where k = min(m,n). Each H_i has the form
 * \[
 *          H_i = I - tau * v * v',
 * \]
 * where tau is a scalar, and v is a vector with
 * \[
 *          v[0] = v[1] = ... = v[i-1] = 0; v[i] = 1,
 * \]
 * with v[i+1] through v[m-1] stored on exit below the diagonal
 * in the ith column of A, and tau in tau[i].
 *
 * @return  0 if success
 *
 * @param[in,out] A m-by-n matrix.
 *      On exit, the elements on and above the diagonal of the array
 *      contain the min(m,n)-by-n upper trapezoidal matrix R
 *      (R is upper triangular if m >= n); the elements below the diagonal,
 *      with the array tau, represent the unitary matrix Q as a
 *      product of elementary reflectors.
 *
 * @param[out] tau Real vector of length min(m,n).
 *      The scalar factors of the elementary reflectors.
 *
 * @ingroup alloc_workspace
 */
template <TLAPACK_SMATRIX matrix_t, TLAPACK_VECTOR vector_t>
int geqr2(matrix_t& A, vector_t& tau)
{
    using idx_t = size_type<matrix_t>;
    using T = type_t<matrix_t>;

    // Functor
    Create<matrix_t> new_matrix;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    // quick return
    if (n <= 0 || m <= 0) return 0;

    // Allocates workspace
    WorkInfo workinfo = geqr2_worksize<T>(A, tau);
    std::vector<T> work_;
    auto work = new_matrix(work_, workinfo.m, workinfo.n);

    return geqr2_work(A, tau, work);
}

}  // namespace tlapack

#endif  // TLAPACK_GEQR2_HH
