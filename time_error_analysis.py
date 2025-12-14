import time
import random
import numpy as np

from error_analysis import cos_angle, CosAdd, VectorCosAdd

def check_time(method="Loop"):
    def decorator(func):
        """Function that measures execution time"""
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            print(f"{method} execution time: {end-start:.6f}")
            return result
        return wrapper
    return decorator


def cos_angle(a: np.ndarray, b: np.ndarray) -> float:
    nominator = np.dot(a, b)
    denominator = np.linalg.norm(a)*np.linalg.norm(b)
    return nominator/denominator

@check_time("Loop")
def CosAdd(b:np.ndarray, embed_matrix: np.ndarray, a: int = 83, a_star: int = 155) -> int:
    best_ind, cos_value = 0, float('-inf')

    for row_ind in range(len(embed_matrix)):
        current_value = cos_angle(embed_matrix[row_ind, :], b - embed_matrix[a, :] + embed_matrix[a_star, :])

        if current_value > cos_value:
            cos_value = current_value
            best_ind = row_ind

    return best_ind

@check_time("Numpy")
def VectorCosAdd(b: np.ndarray, embed_matrix: np.ndarray, a: int = 83, a_star: int = 155) -> int:
    # vectors shape = 1 x E
    # dot product of every matrix row with target vector
    target_vector = b - embed_matrix[a, :] + embed_matrix[a_star, :]
    half_dot = embed_matrix * target_vector
    full_dot = half_dot.sum(axis=1)
    #denomenator
    half_norm = np.apply_along_axis(np.linalg.norm, axis=1, arr=embed_matrix)
    full_norm = half_norm*np.linalg.norm(target_vector)
    return np.argmax(full_dot/full_norm)

def time_3cosadd(num_iters: int = 100) -> None:
    """
    Compares times of loop and vector 3cosadd implementations
    """
    # Iterate the specified number of times
    for _ in range(num_iters):
        # Get size values and create matrix
        VOCAB_SIZE, EMBED_DIM = random.randint(1_000_000, 1_200_000), random.randint(100, 500)
        embed_matrix = np.random.default_rng().random(size=(VOCAB_SIZE, EMBED_DIM))

        # Prepare indecies for a, a_star vectors. Prepare b vector.
        a_ind = random.randint(0, VOCAB_SIZE-1)
        a_star_ind = random.randint(0, VOCAB_SIZE-1)
        b = -embed_matrix[a_ind, :] + 0.15

        loop_value = CosAdd(b, embed_matrix, a=a_ind, a_star=a_star_ind)
        print(loop_value)
        vector_value = VectorCosAdd(b, embed_matrix, a=a_ind, a_star=a_star_ind)
        print(vector_value, end='\n\n')


if __name__ == "__main__":
    time_3cosadd(num_iters = 2)