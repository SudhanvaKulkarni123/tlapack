/// @file potrs.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_LEGACY_POTRS_HH__
#define __TLAPACK_LEGACY_POTRS_HH__

#include "lapack/potrs.hpp"

namespace lapack {

/** Apply the Cholesky factorization to solve a linear system.
 * 
 * @see potrs( uplo_t uplo, const matrixA_t& A, matrixB_t& B )
 * 
 * @ingroup posv_computational
 */
template< typename T >
inline int potrs(
    Uplo uplo, idx_t n, idx_t nrhs,
    const T* A, idx_t lda,
    T* B, idx_t ldb )
{
    using blas::internal::colmajor_matrix;

    // Matrix views
    const auto _A = colmajor_matrix<T>( (T*) A, n, n, lda );
          auto _B = colmajor_matrix<T>( B, n, nrhs, ldb );

    if( uplo == Uplo::Upper )
        return potrs( upper_triangle, _A, _B );
    else
        return potrs( lower_triangle, _A, _B );
}

} // lapack

#endif // __TLAPACK_LEGACY_POTRS_HH__