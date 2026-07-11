# scratch-linalg

A from-scratch linear algebra library. Every operation here is
implemented **twice**: once in pure Python (so the mechanics are
fully visible), and once with NumPy (so you can see and measure what
a real numerical library buys you). No `sklearn`, no `numpy.linalg`
shortcuts in the pure-Python versions — everything is derived and
built by hand first.

This repo is the running foundation for later, more advanced work
(PCA, gradient descent, neural network layers, attention) — all of
it reduces to vectors and matrices, which is why those come first.

---

## Why two implementations of everything?

| | Pure Python (`vectors.py`, `matrices.py`) | NumPy (`vectors_numpy.py`, `matrices_numpy.py`) |
|---|---|---|
| **Purpose** | See every loop, every operation, with nothing hidden | See how a production numerical library executes the *same* math |
| **Storage** | Python `list` / `list of lists` | `numpy.ndarray` (contiguous memory + strides) |
| **Speed** | Slow — Python interpreter overhead per element | Fast — compiled C loops, SIMD, BLAS |
| **When you'd really use this** | Never in production — this is for building intuition | Always, in real ML code |

The pure-Python version and the NumPy version implement **identical
math** with **identical method names** on purpose — you should be
able to swap one for the other and get the same answers (and several
tests explicitly check this — see `test_matches_pure_python_version`
in each `*_numpy.py` file).

---

## Repository structure

```
scratch-linalg/
│
├── vectors.py            # Vector class — pure Python, no dependencies
├── vectors_numpy.py       # VectorNP class — NumPy-backed, same interface
│
├── matrices.py            # Matrix class — pure Python, no dependencies
├── matrices_numpy.py       # MatrixNP class — NumPy-backed, same interface
│
├── eigen.py                # Eigenvalues/eigenvectors — pure Python
├── eigen_numpy.py            # NumPy-backed, plus production np.linalg.eig
│
├── pagerank.py              # PageRank built on eigen.py's power_iteration
│
├── benchmarks/
│   └── benchmark_vectors.py   # Pure Python vs NumPy speed comparison
│
└── README.md              # this file
```

Each file is self-contained: the class definition, followed by its own
`unittest` test suite, runnable directly with:

```bash
python -m unittest vectors.py -v
python -m unittest vectors_numpy.py -v
python -m unittest matrices.py -v
python -m unittest matrices_numpy.py -v
python -m unittest eigen.py -v
python -m unittest eigen_numpy.py -v
python -m unittest pagerank.py -v
```

---

## Architecture: how the pieces relate

```
                     ┌─────────────────────────┐
                     │   Real ML operations      │
                     │  (PCA, neural net layers,  │
                     │   attention)                 │
                     └────────────┬───────────────┘
                                  │ built on top of
                                  ▼
                  ┌───────────────────────────────┐        ┌─────────────────┐
                  │      Eigenvalues/Eigenvectors     │─────>│    pagerank.py     │
                  │  (eigen.py, eigen_numpy.py)         │      │  (applies power_    │
                  │  power_iteration, characteristic     │      │   iteration to a     │
                  │  polynomial, diagonalization           │      │   web graph)           │
                  └───────────────┬───────────────────────┘      └─────────────────┘
                                  │ built on top of
                                  ▼
                  ┌───────────────────────────────┐
                  │           Matrix                │
                  │  (linear transformations)        │
                  │  matmul, matvec, transpose,       │
                  │  trace, identity                  │
                  └───────────────┬───────────────────┘
                                  │ built on top of
                                  ▼
                  ┌───────────────────────────────┐
                  │           Vector                │
                  │   (points/arrows in space)        │
                  │  add, dot, norm, normalize,       │
                  │  cosine_similarity, project_onto   │
                  └───────────────────────────────────┘
```

Conceptually: a **Vector** is a single data point or direction in
space. A **Matrix** is a machine that transforms vectors — and
internally, matrix operations (`matvec`, `matmul`) are built from the
same dot-product logic that `Vector.dot()` uses. This mirrors the real
math: matrix-vector multiplication is just several dot products
bundled together, row by row.

---

## Module reference

### `vectors.py` — `Vector`

| Method | What it does | Complexity |
|---|---|---|
| `__add__`, `__sub__` | component-wise addition/subtraction | O(n) |
| `__mul__` | scalar multiplication (stretch/shrink/flip) | O(n) |
| `dot(other)` | dot product — sum of element-wise products | O(n) |
| `norm()` | Euclidean length, `sqrt(v . v)` | O(n) |
| `normalize()` | returns a unit-length vector in the same direction | O(n) |
| `cosine_similarity(other)` | directional similarity, -1 to 1, magnitude-independent | O(n) |
| `project_onto(other)` | vector projection of self onto other | O(n) |

### `vectors_numpy.py` — `VectorNP`

Same method names and behavior as `Vector`, backed by `np.ndarray`.
Internally uses `np.dot`, `np.linalg.norm` (numerically stable,
scaled internally to avoid overflow/underflow — see code comments)
instead of hand-written loops.

### `matrices.py` — `Matrix`

| Method | What it does | Complexity |
|---|---|---|
| `__add__`, `__sub__` | element-wise addition/subtraction | O(mn) |
| `scalar_mul(c)` | multiply every entry by a scalar | O(mn) |
| `transpose()` | flips rows and columns | O(mn) (copy) |
| `matvec(x)` | matrix-vector multiply, `A @ x` | O(mn) |
| `matmul(other)` | matrix-matrix multiply, `A @ B` | O(mnp), O(n³) for square |
| `identity(n)` | static constructor for the n×n identity matrix | O(n²) |
| `trace()` | sum of diagonal entries (square matrices only) | O(n) |

### `matrices_numpy.py` — `MatrixNP`

Same method names and behavior as `Matrix`, backed by `np.ndarray`.
`transpose()` is effectively **O(1)** here (NumPy relabels memory
strides instead of copying data — see code comments for why), unlike
the pure-Python version which must physically rebuild the grid.

### `eigen.py` — pure Python

| Function | What it does | Complexity |
|---|---|---|
| `characteristic_poly_2x2(A)` | derives `lambda^2 - trace*lambda + det = 0` coefficients | O(1) |
| `solve_quadratic(a,b,c)` | quadratic formula; raises clearly if roots are complex | O(1) |
| `eigen_2x2(A)` | full closed-form eigendecomposition, 2×2 only | O(1) |
| `power_iteration(A)` | general iterative method: finds the dominant eigenvalue/eigenvector of any square matrix | O(n² · iterations) |

The closed-form solver only works for 2×2 matrices (no general
closed-form formula exists for eigenvalues of matrices larger than
4×4 — a consequence of the Abel–Ruffini theorem on polynomial roots).
`power_iteration` is the general-purpose method, and is the same core
idea behind Google's original PageRank algorithm.

### `eigen_numpy.py` — NumPy-backed

| Function | What it does | Complexity |
|---|---|---|
| `power_iteration_np(A)` | vectorized version of `power_iteration` — identical algorithm, `A @ v` via BLAS instead of Python loops | O(n² · iterations) |
| `eigen_np(A)` | wraps `np.linalg.eig` (QR-algorithm-based); works for **any** size matrix and returns complex eigenvalues natively instead of raising | O(n³) |
| `diagonalize(A)` | returns `(P, D, P_inv)` such that `A = P @ D @ P_inv` — the practical realization of the diagonalization identity, useful for cheaply computing `A^k` | O(n³) |

Unlike `eigen.py`'s closed-form solver, `eigen_np` doesn't raise an
error on matrices with no real eigenvectors (e.g. rotation matrices)
— it returns complex-valued results directly, which is what a
production numerical library needs to do.

### `pagerank.py` — pure Python, built on `eigen.py`

An applied case study: Google's original ranking algorithm, reduced
to the observation that "importance" is the stationary distribution
of a random walk on the web graph — which is exactly the eigenvector
for eigenvalue 1. Imports `power_iteration` from `eigen.py` directly
and adds **no new eigenvalue logic** — only the matrix construction
needed to model the random-surfer behavior.

| Function | What it does |
|---|---|
| `build_transition_matrix(edges, nodes)` | column-stochastic matrix `M`; each page distributes its vote equally across its outlinks. Dangling nodes (no outlinks) spread their vote uniformly to avoid leaking probability mass |
| `build_google_matrix(M, damping)` | `G = d*M + (1-d)/N * ones(N,N)` — blends link-following with a random jump (default `d=0.85`), guaranteeing (via the Perron-Frobenius theorem) a unique dominant eigenvalue of exactly 1 |
| `pagerank(edges, nodes)` | full pipeline: build `G`, call `power_iteration(G)` from `eigen.py`, rescale the result to sum to 1 as a probability distribution |
| `print_ranked(scores)` | pretty-prints pages sorted by score |

Run directly to see it rank a 7-node toy graph, cross-checked against
`np.linalg.eig` on the same matrix (exact agreement to 4 decimal
places — see the worked example in `if __name__ == "__main__"`):

```bash
python3 pagerank.py
```

---

## Measured performance difference

From `benchmarks/benchmark_vectors.py` (1,000,000-element vectors) and
a 100×100 matrix multiply benchmark:

| Operation | Pure Python | NumPy | Speedup |
|---|---|---|---|
| Vector dot product (n=1,000,000) | ~60 ms | ~1.5 ms | ~39x |
| Matrix multiply (100×100 @ 100×100) | ~66 ms | ~0.9 ms | ~77x |

The gap grows with size because matrix multiplication is O(n³) — this
is why NumPy/BLAS (and eventually GPUs) aren't a convenience, they're
a requirement once you move past toy examples.

---

## Testing philosophy

Every class ships with a `unittest` suite covering:
- **Known-value tests** — e.g. `dot([1,2,3], [4,5,6]) == 32`, checked
  by hand, not just "runs without crashing."
- **Mathematical property tests** — e.g. transposing twice returns the
  original matrix; matrix multiplication is *not* commutative;
  orthogonal vectors have zero dot product.
- **Edge cases** — dimension mismatches, zero vectors/matrices, all
  expected to raise `ValueError` with a clear message.
- **Cross-implementation tests** — the NumPy version's tests confirm
  it produces the same numbers as the pure-Python version for shared
  inputs, catching any silent divergence between the two.
- **Applied-correctness tests** (`pagerank.py`) — e.g. scores sum to
  1 (it's a probability distribution), the dominant eigenvalue is
  exactly 1 (the Perron-Frobenius guarantee), a symmetric graph
  produces equal ranks for all nodes, and a page with more incoming
  links ranks higher than one with fewer — sanity checks that the
  math is actually modeling the real-world behavior it claims to.

Run everything at once from the repo root:

```bash
python -m unittest discover -p "*.py" -v
```

---

## What comes next

This repo currently covers **Vectors**, **Matrices**,
**Eigenvalues/Eigenvectors**, and one applied case study
(**PageRank**). The natural next addition is **SVD** (Singular Value
Decomposition) — it generalizes eigendecomposition to non-square
matrices and builds directly on `eigen.py`/`eigen_numpy.py`. Looking
further ahead: PCA is built on SVD; a neural network layer is
literally `Matrix.matvec` plus a nonlinearity; attention scores are
`Vector.dot` computed at scale; and the vanishing/exploding gradient
problem in recurrent networks is explained directly by the eigenvalue
spectrum of the weight matrix being applied repeatedly.