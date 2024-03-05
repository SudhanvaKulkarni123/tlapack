#include "tlapack/LegacyMatrix.hpp"
#include "tlapack/base/utils.hpp"


namespace tlapack {

   


template <typename matrix_At, typename real_t>
   void mat_balance(matrix_At& A, Layout L, real_t a) {
    using idx_t = size_type<matrix_At>;
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    int max_exp = 0;
    if(L == Layout::ColMajor){
    for(int i = 0; i < m; i++) {
        max_exp = get_scaling_unit(A(i,0));
         for(int j = 0; j < n; j++){
            //first go through first row, find largest exponent
            if(get_scaling_unit(A(i,j)) > max_exp) {
                max_exp = get_scaling_unit(A(i,j));
            }
         }
         for(int j = 0; j < n; j++) {
            set_float_part(A(i,j),get_float_part(A(i,j))/pow(2.0,max_exp - get_scaling_unit(A(i,j))));
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
            set_float_part(A(i,j),get_float_part(A(i,j))/pow(2.0,max_exp - get_scaling_unit(A(i,j))));
            set_scaling_unit(A(i,j),max_exp);
         }
    }

    }
   }
   }

