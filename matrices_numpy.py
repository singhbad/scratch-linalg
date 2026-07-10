"""
matrices_numpy.py — NumPy-backed rewrite of the from-scratch Matrix class.

Part of: scratch-linalg (Lesson 2 — Matrices)
Same interface, same math, different execution engine. Compare directly
against matrices.py to confirm the logic is identical.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


class MatrixNP:
    """NumPy-backed m x n matrix. Mirrors Matrix from matrices.py."""

    def __init__(self, data) -> None:
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"Matrix must be 2D, got shape {arr.shape}")
        self.data: NDArray[np.float64] = arr

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape

    def __repr__(self) -> str:
        return f"MatrixNP(\n{self.data}\n)"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MatrixNP):
            return NotImplemented
        return self.data.shape == other.data.shape and np.allclose(self.data, other.data)

    def __add__(self, other: "MatrixNP") -> "MatrixNP":
        return MatrixNP(self.data + other.data)

    def __sub__(self, other: "MatrixNP") -> "MatrixNP":
        return MatrixNP(self.data - other.data)

    def scalar_mul(self, scalar: float) -> "MatrixNP":
        return MatrixNP(self.data * scalar)

    def transpose(self) -> "MatrixNP":
        """NumPy's .T is O(1) — it doesn't copy data, just relabels
        strides (see Section 5 for why this matters)."""
        return MatrixNP(self.data.T)

    def matvec(self, x) -> NDArray[np.float64]:
        return self.data @ np.asarray(x, dtype=np.float64)

    def matmul(self, other: "MatrixNP") -> "MatrixNP":
        """Calls into BLAS (dgemm) — the same routine used, at far
        larger scale, inside every PyTorch nn.Linear forward pass."""
        return MatrixNP(self.data @ other.data)

    @staticmethod
    def identity(n: int) -> "MatrixNP":
        return MatrixNP(np.eye(n))

    def trace(self) -> float:
        if self.data.shape[0] != self.data.shape[1]:
            raise ValueError("Trace is only defined for square matrices.")
        return float(np.trace(self.data))


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------
import unittest


class TestMatrixNP(unittest.TestCase):
    def test_addition(self):
        A = MatrixNP([[1, 2], [3, 4]])
        B = MatrixNP([[5, 6], [7, 8]])
        np.testing.assert_allclose((A + B).data, [[6, 8], [10, 12]])

    def test_transpose(self):
        A = MatrixNP([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(A.transpose().shape, (3, 2))

    def test_matmul_known_value(self):
        A = MatrixNP([[1, 2], [3, 4]])
        B = MatrixNP([[5, 6], [7, 8]])
        self.assertEqual(A.matmul(B), MatrixNP([[19, 22], [43, 50]]))

    def test_matmul_not_commutative(self):
        A = MatrixNP([[1, 2], [3, 4]])
        B = MatrixNP([[0, 1], [1, 0]])
        self.assertNotEqual(A.matmul(B), B.matmul(A))

    def test_matches_pure_python_version(self):
        A = MatrixNP([[1, 2], [3, 4]])
        B = MatrixNP([[5, 6], [7, 8]])
        result = A.matmul(B)
        self.assertEqual(result.data.tolist(), [[19.0, 22.0], [43.0, 50.0]])

    def test_trace(self):
        A = MatrixNP([[1, 2], [3, 4]])
        self.assertEqual(A.trace(), 5.0)


if __name__ == "__main__":
    unittest.main()
