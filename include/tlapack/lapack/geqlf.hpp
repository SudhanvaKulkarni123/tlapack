/// @file geqlf.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @note Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/blob/master/SRC/zgeqlf.f
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_GEQLF_HH
#define TLAPACK_GEQLF_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/geql2.hpp"
#include "tlapack/lapack/larfb.hpp"
#include "tlapack/lapack/larft.hpp"

namespace tlapack {
/**
 * Options struct for gelqf
 */
struct GeqlfOpts {
    size_t nb = 32;  ///< Block size
};

/** Worspace query of geqlf()
 *
 * @param[in] A m-by-n matrix.
 *
 * @param tau min(n,m) vector.
 *
 * @param[in] opts Options.
 *
 * @return WorkInfo The amount workspace required.
 *
 * @ingroup workspace_query
 */
template <class T, TLAPACK_SMATRIX A_t, TLAPACK_SVECTOR tau_t>
constexpr WorkInfo geqlf_worksize(const A_t& A,
                                  const tau_t& tau,
                                  const GeqlfOpts& opts = {})
{
    using idx_t = size_type<A_t>;
    using range = pair<idx_t, idx_t>;
    using work_t = matrix_type<A_t, tau_t>;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t k = min(m, n);
    const idx_t nb = min((idx_t)opts.nb, k);

    auto&& A11 = cols(A, range(0, nb));
    auto&& tauw1 = slice(tau, range(0, nb));
    WorkInfo workinfo = geql2_worksize<T>(A11, tauw1);

    if (n > nb) {
        auto&& TT1 = slice(A, range(0, nb), range(0, nb));
        auto&& A12 = slice(A, range(0, m), range(nb, n));
        workinfo.minMax(larfb_worksize<T>(LEFT_SIDE, CONJ_TRANS, BACKWARD,
                                          COLUMNWISE_STORAGE, A11, TT1, A12));
        if constexpr (is_same_v<T, type_t<work_t>>)
            workinfo += WorkInfo(nb, nb);
    }

    return workinfo;
}

/** @copybrief geqlf()
 * Workspace is provided as an argument.
 * @copydetails geqlf()
 *
 * @param work Workspace. Use the workspace query to determine the size needed.
 *
 * @ingroup computational
 */
template <TLAPACK_SMATRIX A_t, TLAPACK_SVECTOR tau_t, TLAPACK_WORKSPACE work_t>
int geqlf_work(A_t& A, tau_t& tau, work_t& work, const GeqlfOpts& opts = {})
{
    using idx_t = size_type<A_t>;
    using range = pair<idx_t, idx_t>;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t k = min(m, n);
    const idx_t nb = min((idx_t)opts.nb, k);

    // check arguments
    tlapack_check((idx_t)size(tau) >= k);

    // Matrix TT
    auto [TT, work2] = (n > nb) ? reshape(work, nb, nb) : reshape(work, 0, 0);

    // Main computational loop
    for (idx_t j2 = 0; j2 < k; j2 += nb) {
        idx_t j = n - j2;
        idx_t ib = min(nb, k - j2);

        // Compute the QR factorization of the current block A(0:m-n+j,j-ib:j)
        auto A11 = slice(A, range(0, m - (n - j)), range(j - ib, j));
        auto tauw1 = slice(tau, range(k - (n - j) - ib, k - (n - j)));

        geql2_work(A11, tauw1, work);

        if (j > ib) {
            // Form the triangular factor of the block reflector
            auto TT1 = slice(TT, range(0, ib), range(0, ib));
            larft(BACKWARD, COLUMNWISE_STORAGE, A11, tauw1, TT1);

            // Apply H to A(0:m-n+j,0:j-ib) from the left
            auto A12 = slice(A, range(0, m - (n - j)), range(0, j - ib));
            larfb_work(LEFT_SIDE, CONJ_TRANS, BACKWARD, COLUMNWISE_STORAGE, A11,
                       TT1, A12, work2);
        }
    }

    return 0;
}

/** Computes an RQ factorization of an m-by-n matrix A using
 *  a blocked algorithm.
 *
 * The matrix Q is represented as a product of elementary reflectors.
 * \[
 *          Q = H(k)**H ... H(2)**H H(1)**H,
 * \]
 * where k = min(m,n). Each H(j) has the form
 * \[
 *          H(j) = I - tauw * w * w**H
 * \]
 * where tauw is a scalar, and w is a vector with
 * \[
 *          w[0] = w[1] = ... = w[j-1] = 0; w[j] = 1,
 * \]
 * where w[j+1]**H through w[n]**H are stored on exit in the jth row of A.
 *
 * @return  0 if success
 *
 * @param[in,out] A m-by-n matrix.
 *      On exit, the elements on and below the diagonal of the array
 *      contain the m by min(m,n) lower trapezoidal matrix L (L is
 *      lower triangular if m <= n); the elements above the diagonal,
 *      with the array tauw, represent the unitary matrix Q as a
 *      product of elementary reflectors.
 *
 * @param[out] tau min(n,m) vector.
 *      The scalar factors of the elementary reflectors.
 *
 * @param[in] opts Options.
 *
 * @ingroup alloc_workspace
 */
template <TLAPACK_SMATRIX A_t, TLAPACK_SVECTOR tau_t>
int geqlf(A_t& A, tau_t& tau, const GeqlfOpts& opts = {})
{
    using work_t = matrix_type<A_t, tau_t>;
    using T = type_t<work_t>;
    Create<work_t> new_matrix;

    // Allocate or get workspace
    WorkInfo workinfo = geqlf_worksize<T>(A, tau, opts);
    std::vector<T> work_;
    auto work = new_matrix(work_, workinfo.m, workinfo.n);

    return geqlf_work(A, tau, work, opts);
}
}  // namespace tlapack
#endif  // TLAPACK_GEQLF_HH
