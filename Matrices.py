"""
matrices.py — A from-scratch Matrix class implementing core linear algebra
operations, built without any external numerical libraries.
 
Part of: scratch-linalg (Lesson 2 — Matrices)
Author: [you]
Purpose: Educational implementation to build first-principles intuition
         for matrix arithmetic before relying on NumPy/PyTorch.
"""
 
from __future__ import annotations
from typing import List
 
 
class Matrix:
    """
    A minimal, explicit implementation of an m x n matrix.
 
    Stored as a list of rows, where each row is a list of floats.
    Every method mirrors an operation you'll later call via NumPy or
    torch.Tensor — the point is to see exactly what's happening
    underneath every nn.Linear layer you'll use starting Month 4.
 
    Attributes:
        rows: number of rows (m)
        cols: number of columns (n)
        data: list of m lists, each of length n
    """
 
    def __init__(self, data: List[List[float]]) -> None:
        if len(data) == 0 or len(data[0]) == 0:
            raise ValueError("Matrix must have at least one row and column.")
        row_len = len(data[0])
        if any(len(row) != row_len for row in data):
            raise ValueError("All rows must have the same length.")
        self.data = [[float(x) for x in row] for row in data]
        self.rows = len(self.data)
        self.cols = row_len
 
    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.cols)
 
    def __repr__(self) -> str:
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix(\n  {rows_str}\n)"
 
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Matrix):
            return NotImplemented
        if self.shape != other.shape:
            return False
        return all(
            abs(self.data[i][j] - other.data[i][j]) < 1e-9
            for i in range(self.rows) for j in range(self.cols)
        )
 
    def __add__(self, other: "Matrix") -> "Matrix":
        """Element-wise addition. Requires identical shape."""
        self._check_same_shape(other)
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])
 
    def __sub__(self, other: "Matrix") -> "Matrix":
        self._check_same_shape(other)
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])
 
    def scalar_mul(self, scalar: float) -> "Matrix":
        return Matrix([[x * scalar for x in row] for row in self.data])
 
    def transpose(self) -> "Matrix":
        """
        Flips rows and columns: (A^T)_ij = A_ji.
        Geometrically: reflects the transformation's row/column roles.
        Used constantly: e.g. A^T A appears in linear regression's
        normal equations and in computing gradients during backprop.
        """
        return Matrix([
            [self.data[i][j] for i in range(self.rows)]
            for j in range(self.cols)
        ])
 
    def matvec(self, x: List[float]) -> List[float]:
        """
        Matrix-vector multiplication: A @ x.
 
        Row i of the result = dot product of row i of A with x.
        Time complexity: O(m*n). Space: O(m) for the output.
        """
        if len(x) != self.cols:
            raise ValueError(
                f"Vector length {len(x)} doesn't match matrix cols {self.cols}"
            )
        return [
            sum(self.data[i][k] * x[k] for k in range(self.cols))
            for i in range(self.rows)
        ]
 
    def matmul(self, other: "Matrix") -> "Matrix":
        """
        Matrix-matrix multiplication: A @ B.
 
        Requires self.cols == other.rows (composing transformations:
        apply B first, then A). Entry (i,j) of result = dot product of
        row i of self with column j of other.
 
        Naive time complexity: O(m*n*p) for (m x n) @ (n x p).
        This is the single most expensive operation in deep learning —
        we discuss faster algorithms and GPU parallelization in Section 5.
        """
        if self.cols != other.rows:
            raise ValueError(
                f"Cannot multiply ({self.rows}x{self.cols}) by "
                f"({other.rows}x{other.cols}): inner dimensions must match."
            )
        result = [[0.0] * other.cols for _ in range(self.rows)]
        for i in range(self.rows):
            for j in range(other.cols):
                total = 0.0
                for k in range(self.cols):
                    total += self.data[i][k] * other.data[k][j]
                result[i][j] = total
        return Matrix(result)
 
    def identity(n: int) -> "Matrix":
        """Static-style constructor: n x n identity matrix (I@A = A)."""
        return Matrix([
            [1.0 if i == j else 0.0 for j in range(n)]
            for i in range(n)
        ])
 
    def trace(self) -> float:
        """Sum of diagonal elements. Requires a square matrix."""
        if self.rows != self.cols:
            raise ValueError("Trace is only defined for square matrices.")
        return sum(self.data[i][i] for i in range(self.rows))
 
    def _check_same_shape(self, other: "Matrix") -> None:
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
 
 
# ---------------------------------------------------------------------------
# Unit tests — run with: python -m unittest matrices.py -v
# ---------------------------------------------------------------------------
import unittest
 
 
class TestMatrix(unittest.TestCase):
    def test_shape(self):
        A = Matrix([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(A.shape, (2, 3))
 
    def test_addition(self):
        A = Matrix([[1, 2], [3, 4]])
        B = Matrix([[5, 6], [7, 8]])
        self.assertEqual((A + B).data, [[6.0, 8.0], [10.0, 12.0]])
 
    def test_scalar_mul(self):
        A = Matrix([[1, 2], [3, 4]])
        self.assertEqual(A.scalar_mul(2).data, [[2.0, 4.0], [6.0, 8.0]])
 
    def test_transpose(self):
        A = Matrix([[1, 2, 3], [4, 5, 6]])
        AT = A.transpose()
        self.assertEqual(AT.shape, (3, 2))
        self.assertEqual(AT.data, [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
 
    def test_transpose_twice_is_identity_operation(self):
        A = Matrix([[1, 2], [3, 4], [5, 6]])
        self.assertEqual(A.transpose().transpose(), A)
 
    def test_matvec_identity(self):
        I = Matrix.identity(3)
        x = [7.0, 8.0, 9.0]
        self.assertEqual(I.matvec(x), x)
 
    def test_matvec_known_value(self):
        A = Matrix([[2, 0], [0, 1]])  # stretches x by 2
        result = A.matvec([1, 1])
        self.assertEqual(result, [2.0, 1.0])
 
    def test_matmul_identity(self):
        A = Matrix([[1, 2], [3, 4]])
        I = Matrix.identity(2)
        self.assertEqual(A.matmul(I), A)
 
    def test_matmul_known_value(self):
        A = Matrix([[1, 2], [3, 4]])
        B = Matrix([[5, 6], [7, 8]])
        # [1*5+2*7, 1*6+2*8] = [19, 22]
        # [3*5+4*7, 3*6+4*8] = [43, 50]
        self.assertEqual(A.matmul(B).data, [[19.0, 22.0], [43.0, 50.0]])
 
    def test_matmul_not_commutative(self):
        A = Matrix([[1, 2], [3, 4]])
        B = Matrix([[0, 1], [1, 0]])  # swap matrix
        self.assertNotEqual(A.matmul(B), B.matmul(A))
 
    def test_matmul_dimension_mismatch_raises(self):
        A = Matrix([[1, 2, 3]])       # 1x3
        B = Matrix([[1, 2], [3, 4]])  # 2x2
        with self.assertRaises(ValueError):
            A.matmul(B)
 
    def test_trace(self):
        A = Matrix([[1, 2], [3, 4]])
        self.assertEqual(A.trace(), 5.0)  # 1 + 4
 
    def test_trace_non_square_raises(self):
        A = Matrix([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ValueError):
            A.trace()
 
 
if __name__ == "__main__":
    unittest.main()
