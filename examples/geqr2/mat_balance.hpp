#include "tlapack/LegacyMatrix.hpp"
#include "tlapack/plugins/float8_iee_p.hpp"

namespace tlapack {

   template <typename matrix_At, typename real_t>
   void mat_balance(matrix_At& A, Layout L, real_t a) {

   return 0;
}



   template <typename matrix_At, int p>
   void mat_balance(matrix_At& A, Layout L, block_float8_ieee<p>) {
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    int max_exp = 0;
    if(L == Layout::ColMajor){
    for(int i = 0; i < m; i++) {
        max_exp = A(i,0).scaling_unit;
         for(int j = 0; j < n; j++){
            //first go through first row, find largest exponent
            if(A(i,j).scaling_unit > max_exp) {
                max_exp = A(i,j).scaling_unit;
            }
         }
         for(int j = 0; j < n; j++) {
            A(i,j).float_part = A(i,j).float_part/pow(2.0,max_exp - A(i,j).scaling_unit);
            A(i,j).scaling_unit = max_exp;
         }
    }
    } else {
        for(int j = 0; j < n; j++) {
        max_exp = A(0,j).scaling_unit;
         for(int i = 0; i < m; i++){
            //first go through first col, find largest exponent
            if(A(i,j).scaling_unit > max_exp) {
                max_exp = A(i,j).scaling_unit;
            }
         }
         for(int i = 0; i < m; i++) {
            A(i,j).float_part = A(i,j).float_part/pow(2.0,max_exp - A(i,j).scaling_unit);
            A(i,j).scaling_unit = max_exp;
         }
    }

    }
   }
   }

