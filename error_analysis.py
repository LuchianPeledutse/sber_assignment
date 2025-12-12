import numpy as np
from scipy import special

# Softmax implementation

input_array = np.array([1, 2, 3], dtype=np.float32)



def softmax(input: np.ndarray, temperature: float) -> np.ndarray:
    # Find maximum value
    M = input.max()
    return np.exp(input/temperature-M)/np.exp(input/temperature - M).sum()


# 3CosAdd implementation

VOCAB_SIZE = 100
EMBED_DIM = 300

embed_matrix = np.random.default_rng().random(size=(VOCAB_SIZE, EMBED_DIM))

def cos_angle(a: np.ndarray, b: np.ndarray) -> float:
    return a.dot(b)/(np.linalg.norm(a)*np.linalg.norm(b))

def CosAdd(b:np.ndarray, a: int = 0, a_star: int = 53) -> int:
    for row_ind in range(len(embed_matrix)):
        pass


if __name__ == "__main__":
    print(f"My softmax value: {softmax(input_array, 1)}")
    print(f"Scipy softmax value: {special.softmax(input_array)}", end='\n\n')

    print(len(embed_matrix))
