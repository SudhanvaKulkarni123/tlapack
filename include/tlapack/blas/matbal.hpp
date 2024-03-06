#include "tlapack/LegacyMatrix.hpp"
#include "tlapack/base/utils.hpp"


namespace tlapack {

   


template <typename matrix_At, typename real_t>
   void mat_balance(matrix_At& A, Layout L, real_t a, bool is_tri, Uplo uplo) {
    using idx_t = size_type<matrix_At>;
    using TA = type_t<matrix_At>;
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    int max_exp = 0;

    if(L == Layout::ColMajor){
      if(is_tri && uplo == Uplo::Upper){
         for(int i = 0; i < m;i++) {
            max_exp = get_scaling_unit(A(i,0));
            for(int j = 0; j < i; j++ ) {
               //go through columns of U forward
               if(get_scaling_unit(A(i,j)) > max_exp) {
                  max_exp = get_scaling_unit(A(i,j));
               }
            }
            for(int j = 0; j < i ; j++) {
               set_float_part(A(i,j),float(get_float_part(A(i,j)))/pow(2.0,max_exp - get_scaling_unit(A(i,j))));
               set_scaling_unit(A(i,j),max_exp);
            }

         }

      } else if(is_tri && uplo == Uplo::Lower){
         for(int i = 0; i < m;i++) {
            max_exp = get_scaling_unit(A(i,n-1));
            for(int j = n-1; j > i-1; j-- ) {
               //go through columns of L backward
               if(get_scaling_unit(A(i,j)) > max_exp) {
                  max_exp = get_scaling_unit(A(i,j));
               }
            }
            for(int j = n-1; j > i-1; j--) {
               set_float_part(A(i,j),float(get_float_part(A(i,j)))/pow(2.0,max_exp - get_scaling_unit(A(i,j))));
               set_scaling_unit(A(i,j),max_exp);
            }

         }

      } else {
      for(int i = 0; i < m; i++) {
         max_exp = get_scaling_unit(A(i,0));
            for(int j = 0; j < n; j++){
               //first go through first row, find largest exponent
               if(get_scaling_unit(A(i,j)) > max_exp) {
                  max_exp = get_scaling_unit(A(i,j));
               }
            }
            for(int j = 0; j < n; j++) {
               set_float_part(A(i,j),float(get_float_part(A(i,j)))/pow(2.0,max_exp - get_scaling_unit(A(i,j))));
               set_scaling_unit(A(i,j),max_exp);
            }
      }
      }
    } else {
      if(is_tri && uplo == Uplo::Upper){
         for(int j = 0; j < n; j++) {
         max_exp = get_scaling_unit(A(m-1,j));
         for(int i = j; i < m; i++){
            //first go through first col, find largest exponent
            if(get_scaling_unit(A(i,j)) > max_exp) {
                max_exp = get_scaling_unit(A(i,j));
            }
         }
         for(int i = j; i < m; i++) {
            set_float_part(A(i,j),float(get_float_part(A(i,j)))/pow(2.0,max_exp - get_scaling_unit(A(i,j))));
            set_scaling_unit(A(i,j),max_exp);
         }
    }
      } else if(is_tri && uplo == Uplo::Lower){
         for(int j = 0; j < n; j++) {
         max_exp = get_scaling_unit(A(0,j));
         for(int i = 0; i < j; i++){
            //first go through first col, find largest exponent
            if(get_scaling_unit(A(i,j)) > max_exp) {
                max_exp = get_scaling_unit(A(i,j));
            }
         }
         for(int i = 0; i < j; i++) {
            set_float_part(A(i,j),float(get_float_part(A(i,j)))/pow(2.0,max_exp - get_scaling_unit(A(i,j))));
            set_scaling_unit(A(i,j),max_exp);
         }
    }

      } else {
        for(int j = 0; j < n; j++) {
        max_exp = get_scaling_unit(A(0,j));
         for(int i = 0; i < m; i++){
            //first go through first col, find largest exponent
            if(get_scaling_unit(A(i,j)) > max_exp) {
                max_exp = get_scaling_unit(A(i,j));
            }
         }
         for(int i = 0; i < m; i++) {
            set_float_part(A(i,j),float(get_float_part(A(i,j)))/pow(2.0,max_exp - get_scaling_unit(A(i,j))));
            set_scaling_unit(A(i,j),max_exp);
         }
    }

    }
   }
   }
}

