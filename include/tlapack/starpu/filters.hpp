/// @file starpu/filters.hpp
/// @brief Filters for StarPU data interfaces.
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_STARPU_FILTERS_HH
#define TLAPACK_STARPU_FILTERS_HH

#include <starpu.h>

namespace tlapack {
namespace starpu {

    using idx_t = uint32_t;

    inline void filter_tile(void* father_interface,
                            void* child_interface,
                            struct starpu_data_filter* f,
                            STARPU_ATTRIBUTE_UNUSED unsigned id,
                            STARPU_ATTRIBUTE_UNUSED unsigned nparts)
    {
        struct starpu_matrix_interface* matrix_father =
            (struct starpu_matrix_interface*)father_interface;
        struct starpu_matrix_interface* matrix_child =
            (struct starpu_matrix_interface*)child_interface;

        idx_t* aux = (idx_t*)(f->filter_arg_ptr);
        idx_t row0 = aux[0];
        idx_t col0 = aux[1];
        idx_t nrows = aux[2];
        idx_t ncols = aux[3];

        matrix_child->id = matrix_father->id;
        matrix_child->elemsize = matrix_father->elemsize;
        matrix_child->nx = nrows;
        matrix_child->ny = ncols;

        /* is the information on this node valid ? */
        if (matrix_father->dev_handle) {
            matrix_child->dev_handle = matrix_father->dev_handle;
            matrix_child->ld = matrix_father->ld;

            const idx_t offset =
                (row0 + col0 * matrix_child->ld) * matrix_child->elemsize;
            if (matrix_father->ptr)
                matrix_child->ptr = matrix_father->ptr + offset;
            matrix_child->offset = matrix_father->offset + offset;

            matrix_child->allocsize =
                matrix_child->ld * matrix_child->ny * matrix_child->elemsize;
        }
        else
            matrix_child->allocsize =
                matrix_child->nx * matrix_child->ny * matrix_child->elemsize;
    }

    inline void filter_ntiles(void* father_interface,
                              void* child_interface,
                              struct starpu_data_filter* f,
                              unsigned id,
                              STARPU_ATTRIBUTE_UNUSED unsigned nparts)
    {
        struct starpu_data_filter f_tile {
            .filter_arg_ptr = (void*)((idx_t*)f->filter_arg_ptr + 4 * id),
        };
        filter_tile(father_interface, child_interface, &f_tile, 0, 1);
    }

    inline void filter_rows(
        void* father_interface,
        void* child_interface,
        STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter* f,
        unsigned id,
        unsigned nchunks)
    {
        struct starpu_matrix_interface* matrix_father =
            (struct starpu_matrix_interface*)father_interface;
        struct starpu_matrix_interface* matrix_child =
            (struct starpu_matrix_interface*)child_interface;

        const idx_t mt = f->filter_arg;

        matrix_child->id = matrix_father->id;
        matrix_child->elemsize = matrix_father->elemsize;
        matrix_child->nx =
            (id == nchunks - 1) ? (matrix_father->nx - (mt * id)) : mt;
        matrix_child->ny = matrix_father->ny;

        /* is the information on this node valid ? */
        if (matrix_father->dev_handle) {
            matrix_child->dev_handle = matrix_father->dev_handle;
            matrix_child->ld = matrix_father->ld;

            const idx_t offset = matrix_child->elemsize * (mt * id);
            if (matrix_father->ptr)
                matrix_child->ptr = matrix_father->ptr + offset;
            matrix_child->offset = matrix_father->offset + offset;

            matrix_child->allocsize =
                matrix_child->ld * matrix_child->ny * matrix_child->elemsize;
        }
        else
            matrix_child->allocsize =
                matrix_child->nx * matrix_child->ny * matrix_child->elemsize;
    }

    inline void filter_cols(void* father_interface,
                            void* child_interface,
                            struct starpu_data_filter* f,
                            unsigned id,
                            unsigned nchunks)
    {
        struct starpu_matrix_interface* matrix_father =
            (struct starpu_matrix_interface*)father_interface;
        struct starpu_matrix_interface* matrix_child =
            (struct starpu_matrix_interface*)child_interface;

        const idx_t nt = f->filter_arg;

        matrix_child->id = matrix_father->id;
        matrix_child->elemsize = matrix_father->elemsize;
        matrix_child->nx = matrix_father->nx;
        matrix_child->ny =
            (id == nchunks - 1) ? (matrix_father->ny - (nt * id)) : nt;

        /* is the information on this node valid ? */
        if (matrix_father->dev_handle) {
            matrix_child->dev_handle = matrix_father->dev_handle;
            matrix_child->ld = matrix_father->ld;

            const idx_t offset =
                matrix_child->elemsize * matrix_child->ld * (nt * id);
            if (matrix_father->ptr)
                matrix_child->ptr = matrix_father->ptr + offset;
            matrix_child->offset = matrix_father->offset + offset;

            matrix_child->allocsize =
                matrix_child->ld * matrix_child->ny * matrix_child->elemsize;
        }
        else
            matrix_child->allocsize =
                matrix_child->nx * matrix_child->ny * matrix_child->elemsize;
    }

}  // namespace starpu
}  // namespace tlapack

#endif  // TLAPACK_STARPU_FILTERS_HH