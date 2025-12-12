import numpy as np
from scipy import special

# Softmax implementation

input_array = np.array([1, 2, 3], dtype=np.float32)



def softmax(input: np.ndarray, temperature: float) -> np.ndarray:
    # Find maximum value
    M = input.max()
    return np.exp(input/temperature-M)/np.exp(input/temperature - M).sum()


# 3CosAdd implementation

VOCAB_SIZE = 200
EMBED_DIM = 300

#random matricies and vectors
embed_matrix = np.random.randint(100, 1_000_000, size=(VOCAB_SIZE, EMBED_DIM)).astype(np.float32)

random_vector = -embed_matrix[155, :]

def cos_angle(a: np.ndarray, b: np.ndarray) -> float:
    nominator = np.dot(a, b)
    denominator = np.linalg.norm(a)*np.linalg.norm(b)
    return nominator/denominator if denominator != 0 else 0.0

def CosAdd(b:np.ndarray, a: int = 83, a_star: int = 155) -> int:
    best_ind, cos_value = 0, float('-inf')

    for row_ind in range(len(embed_matrix)):
        current_value = cos_angle(embed_matrix[row_ind, :], b - embed_matrix[a, :] + embed_matrix[a_star, :])

        if current_value > cos_value:
            cos_value = current_value
            best_ind = row_ind

    return best_ind

def 

if __name__ == "__main__":
    print(f"My softmax value: {softmax(input_array, 1)}")
    print(f"Scipy softmax value: {special.softmax(input_array)}", end='\n\n')

    print(f"Shape of vector we search for: {random_vector.shape}")
    print(f"Answer with python loop: {CosAdd(random_vector)}")
