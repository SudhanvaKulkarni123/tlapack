#include "tlapack/LegacyVector.hpp"
#include "tlapack/base/utils.hpp"


namespace tlapack {

   template <typename vector_Xt, typename real_t>
   void vec_balance(vector_Xt& X, real_t a) {

   return 0;
}



   template <typename vector_Xt, int p>
   void mat_balance(vector_Xt& X, block_float8_ieee<p>) {
    const idx_t m = X.n;


    int max_exp = 0;
    
        max_exp = A[0].scaling_unit;
         for(int j = 0; j < m; j++){
            //first go through first row, find largest exponent
            if(A[j].scaling_unit > max_exp) {
                max_exp = A[j].scaling_unit;
            }
         }
         for(int j = 0; j < n; j++) {
            A[j].float_part = A[j].float_part/pow(2.0,max_exp - A[j].scaling_unit);
            A[j].scaling_unit = max_exp;
         }
    
   

    
   }
   }

