"""
vectors.py — A from-scratch Vector class implementing core linear algebra
operations, built without any external numerical libraries.

Author: Badal Singh
Purpose: Educational implementation to build first-principles intuition
         for vector arithmetic before relying on NumPy/PyTorch.
"""


from __future__ import annotations
import math
from typing import List

class Vector:
    """
    A minimal, explicit implementation of an n-dimensional vector.

    Stored as a plain Python list of floats. Every method here mirrors
    an operation you will later call via NumPy or torch.Tensor — the
    point is to see exactly what's happening underneath.

    Attributes:
        components: the list of scalar values defining the vector.
    """

    def __init__(self, components: List[float]) -> None:
        if len(components) == 0:
            raise ValueError("Vector must have atleast one component.")
        self.components = [ float(c) for c in components ]

    def __len__(self) -> int:
        return len(self.components)
    
    def __repr__(self):
        return f"Vector({self.components})"
    
    def __add__(self, other: "Vector") -> "Vector":
        """Component-wise addition. O(n) time, O(n) space."""
        self._check_same_dimension(other)
        return Vector([a + b for a, b in zip(self.components, other.components)])
    
    def __sub__(self, other: "Vector") -> "Vector":
        """Component-wise subtraction."""
        self._check_same_dimension(other)
        return Vector([a - b for a, b in zip(self.components, other.components)])

    def __mul__(self, scalar: float) -> "Vector":
        """Scalar multiplication: stretches/shrinks/flips the vector."""
        return Vector([c * scalar for c in self.components])

    __rmul__ = __mul__  # allows `2 * v` as well as `v * 2`

    def dot(self, other: "Vector") -> float:
        """
        Dot product: sum of element-wise products.

        Geometrically equals ||u|| * ||v|| * cos(theta).
        Time complexity: O(n). Space complexity: O(1) beyond input.
        """
        self._check_same_dimension(other)
        return sum(a * b for a, b in zip(self.components, other.components))

    def norm(self) -> float:
        """
        Euclidean (L2) length of the vector: sqrt(v . v).
        This is the Pythagorean theorem generalized to n dimensions.
        """
        return math.sqrt(self.dot(self))

    def normalize(self) -> "Vector":
        """
        Returns a unit vector (length 1) pointing in the same direction.
        Used constantly in ML: cosine similarity, weight initialization,
        gradient clipping.
        """
        n = self.norm()
        if n == 0:
            raise ValueError("Cannot normalize the zero vector.")
        return self * (1.0 / n)

    def cosine_similarity(self, other: "Vector") -> float:
        """
        Measures directional similarity between two vectors, independent
        of their magnitude. Ranges from -1 (opposite) to 1 (identical
        direction). This is the core operation behind embedding search,
        recommendation systems, and (later) attention mechanisms.
        """
        denom = self.norm() * other.norm()
        if denom == 0:
            raise ValueError("Cosine similarity undefined for zero vector.")
        return self.dot(other) / denom

    def _check_same_dimension(self, other: "Vector") -> None:
        if len(self) != len(other):
            raise ValueError(
                f"Dimension mismatch: {len(self)} vs {len(other)}"
            )


# ---------------------------------------------------------------------------
# Unit tests — run with: python -m pytest vectors.py  (or use unittest)
# ---------------------------------------------------------------------------
import unittest


class TestVector(unittest.TestCase):
    def test_addition(self):
        u = Vector([1, 2, 3])
        v = Vector([4, 5, 6])
        self.assertEqual((u + v).components, [5.0, 7.0, 9.0])

    def test_dot_product_orthogonal_is_zero(self):
        u = Vector([1, 0])
        v = Vector([0, 1])
        self.assertAlmostEqual(u.dot(v), 0.0)

    def test_norm_pythagorean(self):
        v = Vector([3, 4])
        self.assertAlmostEqual(v.norm(), 5.0)  # classic 3-4-5 triangle

    def test_cosine_similarity_identical_direction(self):
        u = Vector([2, 0])
        v = Vector([5, 0])
        self.assertAlmostEqual(u.cosine_similarity(v), 1.0)

    def test_cosine_similarity_opposite_direction(self):
        u = Vector([1, 0])
        v = Vector([-1, 0])
        self.assertAlmostEqual(u.cosine_similarity(v), -1.0)

    def test_dimension_mismatch_raises(self):
        u = Vector([1, 2])
        v = Vector([1, 2, 3])
        with self.assertRaises(ValueError):
            u + v


if __name__ == "__main__":
    unittest.main()
