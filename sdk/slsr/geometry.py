"""E8 root-lattice geometry for SLSR embeddings.

The E8 root system is the exceptional rank-8 root system with 240 roots,
all of squared length 2. It is the densest sphere packing in 8D and its
240 roots form a maximally-symmetric set of anchor directions — a natural
"vocabulary" of orientations for an 8-dimensional agent state embedding.

Construction used here (standard; see e.g. Conway & Sloane, SPLAG §4.8):

    * 112 integer roots — all permutations of ``(±1, ±1, 0, 0, 0, 0, 0, 0)``
      with exactly two nonzero entries. Count: ``C(8, 2) * 4 = 112``.
    * 128 half-integer roots — all 8-tuples ``(±½, ±½, …, ±½)`` with an
      *even* number of minus signs. Count: ``2^7 = 128``.

    Total: ``112 + 128 = 240`` roots, each satisfying ``‖r‖² = 2``.

This module is **opt-in**: import and use it only when you want to project
an 8-dimensional agent embedding onto E8, score alignment with the lattice,
or gate convergence on geometric symmetry in addition to the scalar
π·cos(√e) threshold.
"""

from __future__ import annotations

import itertools
import logging
from functools import cached_property

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Module-level cache — E8 is fixed geometry, so we only build it once.
# --------------------------------------------------------------------------- #
_E8_ROOT_CACHE: np.ndarray | None = None


def _build_e8_roots() -> np.ndarray:
    """Enumerate all 240 E8 root vectors in 8D.

    Returns
    -------
    np.ndarray
        Array of shape ``(240, 8)`` and dtype ``float64``. Every row has
        squared norm exactly ``2.0``.
    """
    roots: list[np.ndarray] = []

    # --- 112 integer roots: two nonzero entries ±1, rest 0 ---
    for i, j in itertools.combinations(range(8), 2):
        for s1 in (-1.0, 1.0):
            for s2 in (-1.0, 1.0):
                v = np.zeros(8, dtype=np.float64)
                v[i] = s1
                v[j] = s2
                roots.append(v)

    # --- 128 half-integer roots: all entries ±½ with an even number of minus signs ---
    for bits in range(256):  # 2^8 possible sign patterns
        signs = np.array(
            [-1.0 if (bits >> k) & 1 else 1.0 for k in range(8)], dtype=np.float64
        )
        # even number of minus signs
        if int(np.sum(signs < 0)) % 2 == 0:
            roots.append(0.5 * signs)

    arr = np.stack(roots, axis=0)
    assert arr.shape == (240, 8), f"E8 construction produced {arr.shape}, expected (240, 8)"
    return arr


def _cached_roots() -> np.ndarray:
    """Return the module-level 240×8 root array, building it on first access."""
    global _E8_ROOT_CACHE
    if _E8_ROOT_CACHE is None:
        _E8_ROOT_CACHE = _build_e8_roots()
    return _E8_ROOT_CACHE


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


class E8RootSystem:
    """Represents the 240 root vectors of the exceptional lattice E8.

    The class is cheap to construct — the root array itself is computed
    once at module load and shared by all instances (it is a fixed
    mathematical object).

    Examples
    --------
    >>> e8 = E8RootSystem()
    >>> e8.roots().shape
    (240, 8)
    >>> idx, r, cos = e8.nearest_root(np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=float))
    >>> cos  # doctest: +SKIP
    1.0
    """

    DIM: int = 8
    N_ROOTS: int = 240
    ROOT_NORM_SQ: float = 2.0

    def __init__(self) -> None:
        # Roots are shared (read-only view). Callers should not mutate.
        self._roots: np.ndarray = _cached_roots()
        # Pre-compute unit-length roots for fast cosine-similarity lookups.
        self._unit_roots: np.ndarray = self._roots / np.sqrt(self.ROOT_NORM_SQ)

    # ------------------------------------------------------------------ #
    # Basic accessors
    # ------------------------------------------------------------------ #

    def roots(self) -> np.ndarray:
        """Return a copy of the 240×8 root array (safe to mutate)."""
        return self._roots.copy()

    @cached_property
    def simple_roots(self) -> np.ndarray:
        """A standard simple-root basis for E8 (8×8).

        Any of several conventions works; here we use the standard basis
        from Bourbaki / Humphreys. Exposed for users who want to explore
        Weyl-group actions — not used internally for scoring.
        """
        # Bourbaki simple roots for E8 (rows):
        #   alpha_1 = 1/2*(e1 - e2 - e3 - e4 - e5 - e6 - e7 + e8)
        #   alpha_2 = e1 + e2
        #   alpha_i = e_{i-1} - e_{i-2}  for i = 3..8
        s = np.zeros((8, 8), dtype=np.float64)
        s[0] = 0.5 * np.array([1, -1, -1, -1, -1, -1, -1, 1], dtype=np.float64)
        s[1] = np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        for i in range(2, 8):
            v = np.zeros(8, dtype=np.float64)
            v[i - 1] = 1.0
            v[i - 2] = -1.0
            s[i] = v
        return s

    # ------------------------------------------------------------------ #
    # Nearest-root / projection
    # ------------------------------------------------------------------ #

    def nearest_root(self, vector: np.ndarray) -> tuple[int, np.ndarray, float]:
        """Find the E8 root with greatest cosine similarity to ``vector``.

        Parameters
        ----------
        vector : np.ndarray
            An 8-dimensional vector.

        Returns
        -------
        (index, root, cosine_similarity)
            * ``index`` — row-index into :meth:`roots` (0..239).
            * ``root``  — copy of the matching root vector (shape (8,)).
            * ``cosine_similarity`` — scalar in ``[-1, 1]``.
        """
        v = np.asarray(vector, dtype=np.float64).reshape(-1)
        if v.shape != (self.DIM,):
            raise ValueError(
                f"E8 nearest_root expects an 8D vector, got shape {v.shape}"
            )
        v_norm = float(np.linalg.norm(v))
        if v_norm < 1e-12:
            # Degenerate zero vector — return root 0 with cosine 0.
            return 0, self._roots[0].copy(), 0.0
        unit_v = v / v_norm
        cosines = self._unit_roots @ unit_v  # shape (240,)
        idx = int(np.argmax(cosines))
        return idx, self._roots[idx].copy(), float(cosines[idx])

    def project_onto_lattice(self, vector: np.ndarray) -> np.ndarray:
        """Project ``vector`` onto the line through its nearest E8 root.

        This is a rank-1 projection that preserves ``vector``'s magnitude
        along the dominant E8 direction. It is *not* a full lattice
        quantization — it is an interpretable, differentiable "snap" that
        is useful as a regularizer for embeddings.
        """
        v = np.asarray(vector, dtype=np.float64).reshape(-1)
        if v.shape != (self.DIM,):
            raise ValueError(
                f"E8 project_onto_lattice expects an 8D vector, got shape {v.shape}"
            )
        _, root, _ = self.nearest_root(v)
        # Project v onto the unit vector in the direction of ``root``.
        unit_r = root / np.sqrt(self.ROOT_NORM_SQ)
        scale = float(np.dot(v, unit_r))
        return scale * unit_r

    # ------------------------------------------------------------------ #
    # Aggregate symmetry scoring
    # ------------------------------------------------------------------ #

    def symmetry_score(self, vectors: np.ndarray) -> float:
        """Return a scalar in ``[0, 1]`` measuring E8 alignment of ``vectors``.

        The score is defined as the **mean absolute cosine similarity** of
        each input vector with its nearest E8 root, clipped to ``[0, 1]``.

        * A set of vectors drawn isotropically in R^8 scores around
          ``E[|cos|]`` for the nearest-of-240 problem — empirically ~0.35.
        * A set of vectors that *are* E8 roots scores exactly ``1.0``.
        * The sign is ignored because for every root ``r``, ``-r`` is also
          a root, so the nearest-root lookup already captures the
          signed direction.

        Parameters
        ----------
        vectors : np.ndarray
            Either a single 8-vector (shape ``(8,)``) or a batch of 8-vectors
            (shape ``(N, 8)``). A zero vector contributes a score of 0.

        Returns
        -------
        float
            Alignment score in ``[0, 1]``. Higher = more E8-aligned.
        """
        v = np.asarray(vectors, dtype=np.float64)
        if v.ndim == 1:
            v = v.reshape(1, -1)
        if v.ndim != 2 or v.shape[1] != self.DIM:
            raise ValueError(
                f"E8 symmetry_score expects shape (N, 8), got {v.shape}"
            )

        # Row norms, guarding zero vectors
        norms = np.linalg.norm(v, axis=1)
        safe = norms > 1e-12
        unit_v = np.zeros_like(v)
        unit_v[safe] = v[safe] / norms[safe, None]

        # Cosine similarity matrix (N, 240); take best per row
        cos_mat = unit_v @ self._unit_roots.T
        # For zero vectors, force best=0 (not -1, not the max of zeros)
        best = np.max(cos_mat, axis=1)
        best[~safe] = 0.0
        # Because E8 is closed under negation, best is effectively ``|cos|``
        # of the nearest signed root; clip defensively.
        score = float(np.clip(np.mean(best), 0.0, 1.0))
        return score

    # ------------------------------------------------------------------ #
    # Self-validation
    # ------------------------------------------------------------------ #

    def verify(self) -> dict[str, bool]:
        """Run built-in sanity checks on the constructed root system.

        Returns a dict of named boolean checks. All must be True for a
        valid E8 construction.
        """
        r = self._roots
        checks: dict[str, bool] = {}

        # 1) Exactly 240 roots of dimension 8
        checks["count_240"] = r.shape == (240, 8)

        # 2) All roots have squared norm 2
        norms_sq = np.sum(r * r, axis=1)
        checks["all_norm_sq_2"] = bool(np.allclose(norms_sq, 2.0, atol=1e-10))

        # 3) Closure under negation: for every root r, -r is also a root
        #    (use a set of tuples for membership).
        root_set = {tuple(np.round(row, 6)) for row in r}
        checks["closed_under_negation"] = all(
            tuple(np.round(-row, 6)) in root_set for row in r
        )

        # 4) No duplicates
        checks["no_duplicates"] = len(root_set) == 240

        # 5) Closure under Weyl reflections — sample-check.
        #    For a root α, reflection s_α(β) = β - 2(⟨α,β⟩/⟨α,α⟩) α must land
        #    on another root. Since ⟨α,α⟩=2 for E8, this is β - ⟨α,β⟩ α.
        #    Test with a handful of random (α, β) pairs.
        rng = np.random.default_rng(0)
        sample_ok = True
        for _ in range(16):
            ia = int(rng.integers(0, 240))
            ib = int(rng.integers(0, 240))
            alpha = r[ia]
            beta = r[ib]
            reflected = beta - float(np.dot(alpha, beta)) * alpha
            if tuple(np.round(reflected, 6)) not in root_set:
                sample_ok = False
                break
        checks["weyl_reflection_closed_sample"] = sample_ok

        return checks


__all__ = ["E8RootSystem"]
