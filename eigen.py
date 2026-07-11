"""
eigen.py — From-scratch eigenvalue/eigenvector computation.

Part of: scratch-linalg (Lesson 3 — Eigenvalues & Eigenvectors)

Two complementary approaches implemented here:

1. `characteristic_poly_2x2` + `eigen_2x2`: exact, closed-form solution
   for 2x2 matrices via the characteristic polynomial (quadratic
   formula). Only works for 2x2 — doesn't generalize, but makes the
   "det(A - lambda*I) = 0" derivation completely concrete and checkable
   by hand.

2. `power_iteration`: the general, iterative method real numerical
   libraries build on (in more sophisticated forms, e.g. the QR
   algorithm). Finds the dominant (largest-magnitude) eigenvalue and
   its eigenvector for any square matrix, without ever forming the
   characteristic polynomial. This is the same core idea behind
   Google's original PageRank algorithm.
"""

from __future__ import annotations
import math
from typing import List, Tuple

Vec = List[float]
Mat = List[List[float]]


def matvec(A: Mat, x: Vec) -> Vec:
    """Plain matrix-vector multiply (see matrices.py for the full class)."""
    n = len(A)
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(n)]


def vec_norm(x: Vec) -> float:
    return math.sqrt(sum(xi ** 2 for xi in x))


def vec_normalize(x: Vec) -> Vec:
    n = vec_norm(x)
    if n == 0:
        raise ValueError("Cannot normalize the zero vector.")
    return [xi / n for xi in x]


def vec_dot(a: Vec, b: Vec) -> float:
    return sum(ai * bi for ai, bi in zip(a, b))


def characteristic_poly_2x2(A: Mat) -> Tuple[float, float, float]:
    """
    For a 2x2 matrix [[a,b],[c,d]], det(A - lambda*I) expands to:
        lambda^2 - (a+d)*lambda + (a*d - b*c) = 0
    i.e. lambda^2 - trace(A)*lambda + det(A) = 0

    Returns coefficients (1, -trace, det) of this quadratic in lambda.
    This is a genuinely useful shortcut worth memorizing: for ANY 2x2
    matrix, eigenvalues are roots of lambda^2 - trace*lambda + det = 0.
    """
    a, b = A[0]
    c, d = A[1]
    trace = a + d
    det = a * d - b * c
    return (1.0, -trace, det)


def solve_quadratic(a: float, b: float, c: float) -> Tuple[float, float]:
    """
    Standard quadratic formula: roots of a*x^2 + b*x + c = 0.
    Assumes real roots (true whenever A is symmetric — see note below).
    """
    discriminant = b ** 2 - 4 * a * c
    if discriminant < 0:
        raise ValueError(
            "Complex eigenvalues (discriminant < 0) — this matrix "
            "represents a rotation-like transformation with no real "
            "invariant directions. Not handled by this from-scratch "
            "implementation; real libraries return complex numbers here."
        )
    sqrt_disc = math.sqrt(discriminant)
    x1 = (-b + sqrt_disc) / (2 * a)
    x2 = (-b - sqrt_disc) / (2 * a)
    return (x1, x2)


def eigenvector_for_eigenvalue_2x2(A: Mat, lam: float) -> Vec:
    """
    Given a 2x2 matrix A and a known eigenvalue lambda, solve
    (A - lambda*I)v = 0 for v, using the first row of (A - lambda*I).

    For [[p, q], [r, s]] * [v1, v2] = 0:
        p*v1 + q*v2 = 0  =>  v2 = -p/q * v1  (if q != 0)
    Pick v1 = 1 (or handle the q == 0 edge case) then normalize.
    """
    a, b = A[0]
    c, d = A[1]
    p, q = a - lam, b

    if abs(q) > 1e-12:
        v = [1.0, -p / q]
    elif abs(c) > 1e-12:
        # first row was degenerate (q == 0); use the second row instead:
        # c*v1 + (d-lam)*v2 = 0
        r, s = c, d - lam
        v = [1.0, -r / s] if abs(s) > 1e-12 else [0.0, 1.0]
    else:
        v = [1.0, 0.0]  # A - lam*I is diagonal here

    return vec_normalize(v)


def eigen_2x2(A: Mat) -> List[Tuple[float, Vec]]:
    """
    Full closed-form eigendecomposition of a 2x2 matrix.
    Returns [(eigenvalue1, eigenvector1), (eigenvalue2, eigenvector2)].
    """
    a_coef, b_coef, c_coef = characteristic_poly_2x2(A)
    lam1, lam2 = solve_quadratic(a_coef, b_coef, c_coef)
    v1 = eigenvector_for_eigenvalue_2x2(A, lam1)
    v2 = eigenvector_for_eigenvalue_2x2(A, lam2)
    return [(lam1, v1), (lam2, v2)]


def power_iteration(
    A: Mat, num_iterations: int = 100, tol: float = 1e-10
) -> Tuple[float, Vec]:
    """
    Finds the dominant (largest-magnitude) eigenvalue and eigenvector
    of a square matrix A via repeated application + renormalization.

    Algorithm:
        1. Start with a random (here: all-ones) vector v.
        2. Repeat: v = normalize(A @ v)
        3. Once v stabilizes (stops rotating), it's the dominant
           eigenvector. The eigenvalue is recovered via the Rayleigh
           quotient: lambda = (v . Av) / (v . v)  [v is unit length,
           so this simplifies to v . Av].

    Time complexity per iteration: O(n^2) for the matvec.
    Converges linearly, at a rate governed by the ratio of the top
    two eigenvalues |lambda_2 / lambda_1| — the closer this ratio is
    to 1, the slower convergence (this is a real practical concern,
    not just theory: nearly-tied dominant eigenvalues make power
    iteration converge very slowly).
    """
    n = len(A)
    v = vec_normalize([1.0] * n)

    for _ in range(num_iterations):
        v_new = matvec(A, v)
        v_new = vec_normalize(v_new)
        # Convergence check: has the direction stopped changing?
        if vec_norm([v_new[i] - v[i] for i in range(n)]) < tol:
            v = v_new
            break
        v = v_new

    eigenvalue = vec_dot(v, matvec(A, v))  # Rayleigh quotient
    return eigenvalue, v


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------
import unittest


class TestEigen(unittest.TestCase):
    def test_characteristic_poly_2x2_known_matrix(self):
        # A = [[4,1],[2,3]] -> lambda^2 - 7*lambda + 10 = 0
        A = [[4, 1], [2, 3]]
        coeffs = characteristic_poly_2x2(A)
        self.assertEqual(coeffs, (1.0, -7.0, 10.0))

    def test_eigen_2x2_known_eigenvalues(self):
        A = [[4, 1], [2, 3]]
        results = eigen_2x2(A)
        eigenvalues = sorted(lam for lam, _ in results)
        self.assertAlmostEqual(eigenvalues[0], 2.0)
        self.assertAlmostEqual(eigenvalues[1], 5.0)

    def test_eigen_2x2_satisfies_definition(self):
        """The real test: does A @ v actually equal lambda * v?"""
        A = [[4, 1], [2, 3]]
        for lam, v in eigen_2x2(A):
            Av = matvec(A, v)
            lam_v = [lam * vi for vi in v]
            for a, b in zip(Av, lam_v):
                self.assertAlmostEqual(a, b, places=6)

    def test_identity_matrix_eigenvalues_are_one(self):
        I = [[1, 0], [0, 1]]
        results = eigen_2x2(I)
        for lam, _ in results:
            self.assertAlmostEqual(lam, 1.0)

    def test_power_iteration_matches_closed_form(self):
        """Power iteration should find the SAME dominant eigenvalue (5)
        that the exact 2x2 solver found."""
        A = [[4, 1], [2, 3]]
        eigenvalue, eigenvector = power_iteration(A)
        self.assertAlmostEqual(eigenvalue, 5.0, places=4)
        # Eigenvector should satisfy A @ v = lambda * v
        Av = matvec(A, eigenvector)
        for a, b in zip(Av, eigenvector):
            self.assertAlmostEqual(a, eigenvalue * b, places=4)

    def test_power_iteration_diagonal_matrix(self):
        # Dominant eigenvalue of a diagonal matrix is its largest entry
        A = [[7, 0], [0, 3]]
        eigenvalue, _ = power_iteration(A)
        self.assertAlmostEqual(eigenvalue, 7.0, places=4)

    def test_complex_eigenvalues_raise_clear_error(self):
        # A pure rotation matrix (90 degrees) has no real eigenvectors —
        # every vector gets rotated, none just scaled.
        rotation_90 = [[0, -1], [1, 0]]
        with self.assertRaises(ValueError):
            eigen_2x2(rotation_90)


if __name__ == "__main__":
    unittest.main()