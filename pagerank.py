"""
pagerank.py — PageRank from scratch, built directly on eigen.py's
power_iteration function.

Part of: scratch-linalg (Lesson 3 stretch challenge — Eigenvalues in
the wild: Google's original ranking algorithm)

Core idea: model a "random surfer" clicking links forever. A page's
long-run importance is the probability of finding the surfer there —
i.e. the STATIONARY DISTRIBUTION of a random walk on the web graph.
Stationary distributions of a stochastic matrix are exactly the
eigenvector for eigenvalue lambda=1 — so PageRank reduces to finding
a dominant eigenvector, which power_iteration already does.

Reference: Page, Brin, Motwani, Winograd (1998),
"The PageRank Citation Ranking: Bringing Order to the Web."
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import sys
import os

# Reuse the eigen.py power_iteration implementation directly —
# this is the whole point of the exercise: no new eigenvalue logic,
# just a new matrix to apply it to.
sys.path.insert(0, os.path.dirname(__file__))
from eigen import power_iteration, matvec  # noqa: E402


def build_transition_matrix(
    edges: List[Tuple[str, str]], nodes: List[str]
) -> List[List[float]]:
    """
    Builds the column-stochastic transition matrix M for a directed graph.

    M[i][j] = 1 / outdegree(j)  if page j links to page i
              1 / N             if page j is a dangling node (no outlinks) —
                                 spreads its vote uniformly to avoid leaking
                                 probability mass out of the system
              0                 otherwise

    Each COLUMN sums to 1: page j distributes its full "vote" across
    everything it links to (or everything, if it links to nothing).
    """
    n = len(nodes)
    index = {node: i for i, node in enumerate(nodes)}

    outdegree = {node: 0 for node in nodes}
    for src, _ in edges:
        outdegree[src] += 1

    M = [[0.0] * n for _ in range(n)]
    for src, dst in edges:
        j = index[src]
        i = index[dst]
        M[i][j] = 1.0 / outdegree[src]

    # Dangling nodes: no outlinks -> spread vote uniformly over all pages
    for node in nodes:
        if outdegree[node] == 0:
            j = index[node]
            for i in range(n):
                M[i][j] = 1.0 / n

    return M


def build_google_matrix(M: List[List[float]], damping: float = 0.85) -> List[List[float]]:
    """
    G = d*M + (1-d)/N * ones(N,N)

    The random-jump term guarantees G is irreducible and aperiodic
    (Perron-Frobenius conditions), which is what guarantees a UNIQUE
    dominant eigenvalue of exactly 1 with an all-positive eigenvector —
    without this term, some graphs would have no well-defined ranking.
    """
    n = len(M)
    teleport = (1.0 - damping) / n
    return [
        [damping * M[i][j] + teleport for j in range(n)]
        for i in range(n)
    ]


def pagerank(
    edges: List[Tuple[str, str]],
    nodes: List[str],
    damping: float = 0.85,
    num_iterations: int = 200,
) -> Dict[str, float]:
    """
    Full PageRank pipeline: build the graph's Google matrix, find its
    dominant eigenvector via power_iteration (imported unchanged from
    eigen.py), then rescale it to sum to 1 (a valid final step, since
    eigenvectors are only defined up to scale — see Lesson 3, Section
    2 — and here we specifically want a probability distribution).
    """
    M = build_transition_matrix(edges, nodes)
    G = build_google_matrix(M, damping)

    eigenvalue, eigenvector = power_iteration(G, num_iterations=num_iterations)

    # power_iteration can return a vector pointing in either direction
    # along the eigenvector's line (sign ambiguity, see Lesson 3) —
    # flip it positive before turning it into a probability distribution.
    if sum(eigenvector) < 0:
        eigenvector = [-x for x in eigenvector]

    total = sum(eigenvector)
    scores = [x / total for x in eigenvector]

    return dict(zip(nodes, scores)), eigenvalue


def print_ranked(scores: Dict[str, float]) -> None:
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    print(f"{'Rank':<6}{'Page':<8}{'PageRank Score':<16}")
    print("-" * 30)
    for rank, (node, score) in enumerate(ranked, start=1):
        print(f"{rank:<6}{node:<8}{score:.4f}")


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------
import unittest


class TestPageRank(unittest.TestCase):
    def test_transition_matrix_columns_sum_to_one(self):
        nodes = ["A", "B", "C"]
        edges = [("A", "B"), ("A", "C"), ("B", "C"), ("C", "A")]
        M = build_transition_matrix(edges, nodes)
        n = len(nodes)
        for j in range(n):
            col_sum = sum(M[i][j] for i in range(n))
            self.assertAlmostEqual(col_sum, 1.0, places=9)

    def test_dangling_node_handled(self):
        # C has no outlinks -- should spread uniformly, not leak mass
        nodes = ["A", "B", "C"]
        edges = [("A", "B"), ("B", "C")]
        M = build_transition_matrix(edges, nodes)
        col_sum = sum(M[i][2] for i in range(3))  # column for C
        self.assertAlmostEqual(col_sum, 1.0, places=9)

    def test_google_matrix_columns_sum_to_one(self):
        nodes = ["A", "B", "C"]
        edges = [("A", "B"), ("B", "A"), ("B", "C"), ("C", "A")]
        M = build_transition_matrix(edges, nodes)
        G = build_google_matrix(M, damping=0.85)
        for j in range(len(nodes)):
            col_sum = sum(G[i][j] for i in range(len(nodes)))
            self.assertAlmostEqual(col_sum, 1.0, places=6)

    def test_pagerank_scores_sum_to_one(self):
        nodes = ["A", "B", "C", "D"]
        edges = [("A", "B"), ("B", "C"), ("C", "A"), ("C", "D"), ("D", "C")]
        scores, _ = pagerank(edges, nodes)
        self.assertAlmostEqual(sum(scores.values()), 1.0, places=6)

    def test_dominant_eigenvalue_is_one(self):
        # Perron-Frobenius guarantee for a column-stochastic Google matrix
        nodes = ["A", "B", "C"]
        edges = [("A", "B"), ("B", "C"), ("C", "A")]
        _, eigenvalue = pagerank(edges, nodes)
        self.assertAlmostEqual(eigenvalue, 1.0, places=6)

    def test_more_incoming_links_means_higher_rank(self):
        # C is linked to by both A and B; A and B are linked to by no one
        # except each other minimally. C should clearly rank highest.
        nodes = ["A", "B", "C"]
        edges = [("A", "C"), ("B", "C"), ("C", "A")]
        scores, _ = pagerank(edges, nodes)
        self.assertGreater(scores["C"], scores["A"])
        self.assertGreater(scores["C"], scores["B"])

    def test_symmetric_graph_gives_equal_ranks(self):
        # A cycle: A->B->C->A. Perfect symmetry -> all pages equally ranked.
        nodes = ["A", "B", "C"]
        edges = [("A", "B"), ("B", "C"), ("C", "A")]
        scores, _ = pagerank(edges, nodes)
        vals = list(scores.values())
        for v in vals:
            self.assertAlmostEqual(v, 1.0 / 3, places=3)


if __name__ == "__main__":
    # Toy graph: 7 "web pages" with realistic asymmetric linking —
    # this is the graph actually ranked when you run this file directly.
    #
    #     A ──> B ──> C
    #     │     │     │
    #     v     v     v
    #     D <── E ──> F ──> G (dangling: no outlinks)
    #     ^                 │
    #     └─────────────────┘
    #
    nodes = ["A", "B", "C", "D", "E", "F", "G"]
    edges = [
        ("A", "B"),
        ("A", "D"),
        ("B", "C"),
        ("B", "E"),
        ("C", "F"),
        ("D", "E"),
        ("E", "D"),
        ("E", "F"),
        ("F", "G"),
        ("G", "D"),  # dangling-node fix would trigger if this line were removed
    ]

    print("Toy web graph:")
    print("  A -> B, D")
    print("  B -> C, E")
    print("  C -> F")
    print("  D -> E")
    print("  E -> D, F")
    print("  F -> G")
    print("  G -> D")
    print()

    scores, eigenvalue = pagerank(edges, nodes)
    print(f"Dominant eigenvalue found: {eigenvalue:.6f}  (should be ~1.0)\n")
    print_ranked(scores)

    print()
    print("Running unit tests...")
    unittest.main(argv=[sys.argv[0]], exit=False, verbosity=2)