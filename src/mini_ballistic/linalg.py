import numpy as np

def gauss_solve(A: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix of size n x n")

    if b.ndim != 1 or b.shape[0] != A.shape[0]:
        raise ValueError("b must be a vector with the same length as size of A")

    n = len(A)
    scale = np.max(np.abs(A))

    for k in range(n):
        pivot_row = k + int(np.argmax(np.abs(A[k:, k])))

        if abs(A[pivot_row, k]) <= eps * scale:
            raise ValueError(f"A is singular at index {k}")

        if pivot_row != k:
            A[[k, pivot_row]], b[[k, pivot_row]] = A[[pivot_row, k]], b[[pivot_row, k]]

        for i in range(k+1, n): 
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    x = np.zeros(n)
    for k in reversed(range(n)):
        x[k] = (b[k] - A[k, k+1:] @ x[k+1:]) / A[k,k]
            
    return x

def hilbert(size: int) -> np.ndarray:
    H = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            H[i, j] = 1 / (i + j + 1)

    return H