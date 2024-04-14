# \<T\>LAPACK

This is a fork of the original [\<T\>LAPACK] repo and contains some work on testing th effectiveness of 8-bit floats on HouseholderQR (geqr2) and Gaussian Elim w Partial Pivoting (getrf). This is achieve by scaling, mixed precision BLAS, block floating point and other strategies. We are also testing how useful these low-precision solutions can be in Iterative refinement schemes.

## relevant files
\<T\>LAPACK is a large project and not all files/algorithms are required for this study. 
Most of the required work can be found in the include/plugins, include/blas, examples/geqr2 and examples/lu directories 

## License

BSD 3-Clause License

Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.

Copyright (c) 2017-2021, University of Tennessee. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## References

<a id="1">[1]</a> Higham, N. J. (2002). Accuracy and stability of numerical algorithms. Society for industrial and applied mathematics.
