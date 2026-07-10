"""
vectors_numpy.py — NumPy-backed rewrite of the from-scratch Vector class.

Part of: scratch-linalg (Lesson 1 — Vectors)
Same interface, same math, different execution engine underneath.
Compare directly against vectors.py (the pure-Python version) to confirm
that the *logic* is identical — only how each operation executes differs.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


class VectorNP:
    """
    NumPy-backed n-dimensional vector.

    Internally stores components as a 1D np.ndarray instead of a Python
    list. Every method mirrors Vector from vectors.py.
    """

    def __init__(self, components) -> None:
        arr = np.asarray(components, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"Vector must be 1D, got shape {arr.shape}")
        if arr.size == 0:
            raise ValueError("Vector must have at least one component.")
        self.data: NDArray[np.float64] = arr

    def __len__(self) -> int:
        return self.data.shape[0]

    def __repr__(self) -> str:
        return f"VectorNP({self.data.tolist()})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VectorNP):
            return NotImplemented
        return (
            self.data.shape == other.data.shape
            and np.allclose(self.data, other.data)
        )

    def __add__(self, other: "VectorNP") -> "VectorNP":
        self._check_same_dimension(other)
        return VectorNP(self.data + other.data)   # element-wise, in C, not Python

    def __sub__(self, other: "VectorNP") -> "VectorNP":
        self._check_same_dimension(other)
        return VectorNP(self.data - other.data)

    def __mul__(self, scalar: float) -> "VectorNP":
        return VectorNP(self.data * scalar)

    __rmul__ = __mul__

    def dot(self, other: "VectorNP") -> float:
        """Dot product via np.dot — calls into BLAS under the hood."""
        self._check_same_dimension(other)
        return float(np.dot(self.data, other.data))

    def norm(self) -> float:
        """Euclidean length via np.linalg.norm (numerically stable version)."""
        return float(np.linalg.norm(self.data))

    def normalize(self) -> "VectorNP":
        n = self.norm()
        if n == 0:
            raise ValueError("Cannot normalize the zero vector.")
        return VectorNP(self.data / n)

    def cosine_similarity(self, other: "VectorNP") -> float:
        denom = self.norm() * other.norm()
        if denom == 0:
            raise ValueError("Cosine similarity undefined for zero vector.")
        return float(np.dot(self.data, other.data) / denom)

    def project_onto(self, other: "VectorNP") -> "VectorNP":
        """
        Vector projection of self onto other:
            proj = [(self . other) / ||other||^2] * other

        See vectors.py's project_onto docstring for the full derivation —
        identical math, vectorized execution.
        """
        self._check_same_dimension(other)
        denom = float(np.dot(other.data, other.data))
        if denom == 0:
            raise ValueError("Cannot project onto the zero vector.")
        scale = float(np.dot(self.data, other.data)) / denom
        return VectorNP(other.data * scale)

    def _check_same_dimension(self, other: "VectorNP") -> None:
        if len(self) != len(other):
            raise ValueError(f"Dimension mismatch: {len(self)} vs {len(other)}")


# ---------------------------------------------------------------------------
# Unit tests — mirrors vectors.py's test suite exactly, proving equivalence
# ---------------------------------------------------------------------------
import unittest


class TestVectorNP(unittest.TestCase):
    def test_addition(self):
        u = VectorNP([1, 2, 3])
        v = VectorNP([4, 5, 6])
        np.testing.assert_allclose((u + v).data, [5.0, 7.0, 9.0])

    def test_subtraction(self):
        u = VectorNP([5, 7, 9])
        v = VectorNP([4, 5, 6])
        np.testing.assert_allclose((u - v).data, [1.0, 2.0, 3.0])

    def test_scalar_multiplication(self):
        v = VectorNP([1, 2, 3])
        np.testing.assert_allclose((v * 2).data, [2.0, 4.0, 6.0])
        np.testing.assert_allclose((2 * v).data, [2.0, 4.0, 6.0])

    def test_dot_product_orthogonal_is_zero(self):
        u = VectorNP([1, 0])
        v = VectorNP([0, 1])
        self.assertAlmostEqual(u.dot(v), 0.0)

    def test_dot_product_known_value(self):
        u = VectorNP([1, 2, 3])
        v = VectorNP([4, 5, 6])
        self.assertAlmostEqual(u.dot(v), 32.0)

    def test_norm_pythagorean(self):
        v = VectorNP([3, 4])
        self.assertAlmostEqual(v.norm(), 5.0)

    def test_normalize_produces_unit_length(self):
        v = VectorNP([3, 4])
        self.assertAlmostEqual(v.normalize().norm(), 1.0)

    def test_cosine_similarity_identical_direction(self):
        u = VectorNP([2, 0])
        v = VectorNP([5, 0])
        self.assertAlmostEqual(u.cosine_similarity(v), 1.0)

    def test_project_onto_known_value(self):
        u = VectorNP([3, 4])
        v = VectorNP([1, 0])
        proj = u.project_onto(v)
        self.assertEqual(proj, VectorNP([3.0, 0.0]))

    def test_dimension_mismatch_raises(self):
        u = VectorNP([1, 2])
        v = VectorNP([1, 2, 3])
        with self.assertRaises(ValueError):
            u + v

    def test_matches_pure_python_version(self):
        # Cross-check: same inputs, same outputs as vectors.py's Vector class
        u = VectorNP([1, 2, 3])
        v = VectorNP([4, 5, 6])
        self.assertAlmostEqual(u.dot(v), 32.0)
        self.assertAlmostEqual(u.norm(), 3.7416573867739413)


if __name__ == "__main__":
    unittest.main()
