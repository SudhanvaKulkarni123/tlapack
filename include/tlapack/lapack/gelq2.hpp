/// @file gelq2.hpp
/// @author Yuxin Cai, University of Colorado Denver, USA
/// @note Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/blob/master/SRC/zgelq2.f
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_GELQ2_HH
#define TLAPACK_GELQ2_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/larf.hpp"
#include "tlapack/lapack/larfg.hpp"

namespace tlapack {

/** Worspace query of gelq2()
 *
 * @param[in] A m-by-n matrix.
 *
 * @param tauw Not referenced.
 *
 * @param[in] opts Options.
 *
 * @return WorkInfo The amount workspace required.
 *
 * @ingroup workspace_query
 */
template <TLAPACK_SMATRIX matrix_t, TLAPACK_VECTOR vector_t>
inline constexpr WorkInfo gelq2_worksize(const matrix_t& A,
                                         const vector_t& tauw,
                                         const WorkspaceOpts& opts = {})
{
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    // constants
    const idx_t m = nrows(A);

    if (m > 1) {
        auto C = rows(A, range{1, m});
        return larf_worksize(RIGHT_SIDE, FORWARD, ROWWISE_STORAGE, row(A, 0),
                             tauw[0], C, opts);
    }
    return WorkInfo{};
}

/** Computes an LQ factorization of a complex m-by-n matrix A using
 *  an unblocked algorithm.
 *
 * The matrix Q is represented as a product of elementary reflectors.
 * \[
 *          Q = H(k)**H ... H(2)**H H(1)**H,
 * \]
 * where k = min(m,n). Each H(j) has the form
 * \[
 *          H(j) = I - tauw * w * w**H
 * \]
 * where tauw is a complex scalar, and w is a complex vector with
 * \[
 *          w[0] = w[1] = ... = w[j-1] = 0; w[j] = 1,
 * \]
 * with w[j+1]**H through w[n]**H is stored on exit
 * in the jth row of A, and tauw in tauw[j].
 *
 *
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
 * @param[out] tauw Complex vector of length min(m,n).
 *      The scalar factors of the elementary reflectors.
 *
 * @param[in] opts Options.
 *      - @c opts.work is used if whenever it has sufficient size.
 *        The sufficient size can be obtained through a workspace query.
 *
 * @ingroup computational
 */
template <TLAPACK_SMATRIX matrix_t, TLAPACK_VECTOR vector_t>
int gelq2(matrix_t& A, vector_t& tauw, const WorkspaceOpts& opts = {})
{
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t k = min(m, n);

    // check arguments
    tlapack_check_false((idx_t)size(tauw) < min(m, n));

    // Allocates workspace
    VectorOfBytes localworkdata;
    Workspace work = [&]() {
        WorkInfo workinfo = gelq2_worksize(A, tauw, opts);
        return alloc_workspace(localworkdata, workinfo, opts.work);
    }();

    // Options to forward
    auto&& larfOpts = WorkspaceOpts{work};

    for (idx_t j = 0; j < k; ++j) {
        // Define w := A(j,j:n)
        auto w = slice(A, j, range(j, n));

        // Generate elementary reflector H(j) to annihilate A(j,j+1:n)
        larfg(FORWARD, ROWWISE_STORAGE, w, tauw[j]);

        // If either condition is satisfied, Q11 will not be empty
        if (j < k - 1 || k < m) {
            // Apply H(j) to A(j+1:m,j:n) from the right
            auto Q11 = slice(A, range(j + 1, m), range(j, n));
            larf(Side::Right, FORWARD, ROWWISE_STORAGE, w, tauw[j], Q11,
                 larfOpts);
        }
    }

    return 0;
}
}  // namespace tlapack

#endif  // TLAPACK_GELQ2_HH
