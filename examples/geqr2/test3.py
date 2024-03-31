import numpy as np


def generate_random_matrix(n):
    # Generate n-by-n random matrix of floats with float16 datatype
    random_matrix = np.random.rand(n, n).astype(np.float16)
    return random_matrix

