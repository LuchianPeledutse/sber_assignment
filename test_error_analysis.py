import pytest

import random
import numpy as np

from error_analysis import cos_angle, CosAdd, VectorCosAdd

  
@pytest.mark.parametrize(
        "mat_size_values", 
        [(random.randint(100, 10_000), random.randint(3, 500)) for _ in range(100)])
def test_3cosadd(mat_size_values: tuple) -> None:
    """
    Tests equality of python loop 3cosadd and vector 3cosadd
    """
    # Get size values and create matrix
    VOCAB_SIZE, EMBED_DIM = mat_size_values
    embed_matrix = np.random.default_rng().random(size=(VOCAB_SIZE, EMBED_DIM))

    # Prepare indecies for a, a_star vectors. Prepare b vector.
    a_ind = random.randint(0, VOCAB_SIZE-1)
    a_star_ind = random.randint(0, VOCAB_SIZE-1)
    b = -embed_matrix[a_ind, :] + 0.15

    loop_value = CosAdd(b, embed_matrix, a=a_ind, a_star=a_star_ind)
    vector_value = VectorCosAdd(b, embed_matrix, a=a_ind, a_star=a_star_ind)
    assert loop_value == vector_value, "Loop and vector values are not equal" 




