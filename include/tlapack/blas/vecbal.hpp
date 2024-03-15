#include "tlapack/LegacyVector.hpp"
#include "tlapack/base/utils.hpp"


namespace tlapack {

   
   template <typename vector_Xt, typename real_t>
   void vec_balance(vector_Xt& X, real_t a) {
     using idx_t = size_type<vector_Xt>;
     using TX = type_t<vector_Xt>;
    const idx_t m = X.n;
   

    int max_exp = 0;
    
        max_exp = get_scaling_unit(X[0]);
         for(int j = 0; j < m; j++){
            //first go through first row, find largest exponent
            if(get_scaling_unit(X[j]) > max_exp) {
                max_exp = get_scaling_unit(X[j]);
            }
         }
         for(int j = 0; j < m; j++) {
            set_float_part(X[j], get_float_part(X[j])/pow(2.0,max_exp - get_scaling_unit(X[j])));
            set_scaling_unit(X[j],max_exp);
         }
    
   

    
   }
   }

