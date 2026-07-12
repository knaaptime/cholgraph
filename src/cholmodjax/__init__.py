"""cholmodjax: JAX-native sparse Cholesky via CHOLMOD.

Exposes CHOLMOD's sparse Cholesky factorization as XLA FFI custom calls, so
solves run at full native speed inside ``@jax.jit`` (and ``lax.scan`` /
``lax.fori_loop``) with no Python callback overhead.

The matrix is a symmetric positive definite matrix in COO format. Entries on
or above the diagonal (``Ai <= Aj``) are used; entries below the diagonal are
ignored, so you may pass either the full symmetric matrix or just its upper
triangle. Duplicate entries are summed.

Symbolic analysis (fill-reducing ordering + elimination tree) is cached inside
the extension, keyed on the sparsity pattern. Repeated solves with the same
pattern — the typical Gibbs-sampler loop — only pay for the numeric
refactorization, and solves with unchanged values skip even that.

Example::

    import jax, jax.numpy as jnp
    import cholmodjax

    jax.config.update("jax_enable_x64", True)

    @jax.jit
    def step(Ax, b):
        return cholmodjax.solve(Ai, Aj, Ax, b)   # full CHOLMOD speed in JIT

``solve`` is differentiable in ``Ax`` and ``b`` (reverse mode). Under
``jax.vmap`` the whole batch is solved in a single native FFI call that loops
in C++ and reuses the cached analysis. :func:`update_solve` exposes CHOLMOD's
rank-k update/downdate for cheap solves of ``(A ± C C') x = b`` with ``A`` held
fixed.
"""

import jax
import jax.numpy as jnp
import numpy as np

import cholmodjax_cpp as _cpp

__version__ = "0.3.0"
__all__ = [
    "solve",
    "logdet",
    "update_solve",
    "solve_bcoo",
    "logdet_bcoo",
    "update_solve_bcoo",
    "clear_cache",
    "cache_size",
    "set_options",
    "MODE_A",
    "MODE_LDLT",
    "MODE_LD",
    "MODE_DLT",
    "MODE_L",
    "MODE_LT",
    "MODE_D",
    "MODE_P",
    "MODE_PT",
]

jax.ffi.register_ffi_target(
    "cholmodjax_solve_f64", _cpp.solve_f64_capsule(), platform="cpu"
)
jax.ffi.register_ffi_target(
    "cholmodjax_solve_batched_f64",
    _cpp.solve_batched_f64_capsule(),
    platform="cpu",
)
jax.ffi.register_ffi_target(
    "cholmodjax_logdet_f64", _cpp.logdet_f64_capsule(), platform="cpu"
)
jax.ffi.register_ffi_target(
    "cholmodjax_updown_solve_f64",
    _cpp.updown_solve_f64_capsule(),
    platform="cpu",
)

# Solve modes, matching cholmod.h's system codes. MODE_A solves A x = b.
# The factorization is P'LL'P = A, so e.g. sampling from N(0, A^{-1}) is
# solve(..., z, mode=MODE_LT) followed by solve(..., ., mode=MODE_PT).
MODE_A = 0
MODE_LDLT = 1
MODE_LD = 2
MODE_DLT = 3
MODE_L = 4
MODE_LT = 5
MODE_D = 6
MODE_P = 7
MODE_PT = 8


def _require_x64():
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "cholmodjax requires 64-bit mode. Call "
            'jax.config.update("jax_enable_x64", True) before using it.'
        )


def _solve_batched(Ai, Aj, Ax, b, mode):
    # One FFI call for a whole batch (leading axis 0): Ax is (B, nnz), b is
    # (B, n[, nrhs]); the C++ handler loops over B reusing the cached analysis.
    call = jax.ffi.ffi_call(
        "cholmodjax_solve_batched_f64", jax.ShapeDtypeStruct(b.shape, b.dtype)
    )
    return call(Ai, Aj, Ax, b, mode=np.int64(mode))


# One custom_vmap-wrapped dispatcher per solve mode: an ordinary (unbatched)
# FFI call normally, but under vmap it routes to the batched handler so the
# batch loop runs in C++ instead of as XLA per-iteration dispatch. Ai/Aj are
# never mapped (the pattern is shared), so they are not broadcast.
_DISPATCH = {}


def _make_solve_dispatch(mode):
    @jax.custom_batching.custom_vmap
    def dispatch(Ai, Aj, Ax, b):
        call = jax.ffi.ffi_call(
            "cholmodjax_solve_f64", jax.ShapeDtypeStruct(b.shape, b.dtype)
        )
        return call(Ai, Aj, Ax, b, mode=np.int64(mode))

    @dispatch.def_vmap
    def _dispatch_vmap(axis_size, in_batched, Ai, Aj, Ax, b):
        _, _, Ax_batched, b_batched = in_batched
        if not Ax_batched:
            Ax = jnp.broadcast_to(Ax, (axis_size,) + Ax.shape)
        if not b_batched:
            b = jnp.broadcast_to(b, (axis_size,) + b.shape)
        return _solve_batched(Ai, Aj, Ax, b, mode), True

    return dispatch


def _solve_ffi(Ai, Aj, Ax, b, mode):
    dispatch = _DISPATCH.get(mode)
    if dispatch is None:
        dispatch = _DISPATCH[mode] = _make_solve_dispatch(mode)
    return dispatch(Ai, Aj, Ax, b)


def solve(Ai, Aj, Ax, b, mode=MODE_A):
    """Solve ``A x = b`` for symmetric positive definite sparse ``A``.

    Works inside ``@jax.jit``; the sparsity pattern's symbolic analysis is
    computed once and cached across calls.

    Args:
        Ai: ``[n_nz]`` int32 — COO row indices.
        Aj: ``[n_nz]`` int32 — COO column indices. Entries with ``Ai > Aj``
            are ignored (the matrix is taken from the upper triangle).
        Ax: ``[n_nz]`` float64 — COO values. Duplicates are summed.
        b: ``[n]`` or ``[n, n_rhs]`` float64 — right-hand side(s).
        mode: which system to solve, one of the ``MODE_*`` constants.
            ``MODE_A`` (default) solves ``A x = b``; other modes expose the
            factor parts (e.g. ``MODE_LT`` for ``L^T x = b``).

    Returns:
        ``x`` with the same shape as ``b``.

    Raises:
        Exception: if the matrix is not positive definite (raised by the XLA
            runtime with the CHOLMOD failure column in the message).
    """
    _require_x64()
    Ai = jnp.asarray(Ai, jnp.int32)
    Aj = jnp.asarray(Aj, jnp.int32)
    Ax = jnp.asarray(Ax, jnp.float64)
    b = jnp.asarray(b, jnp.float64)
    if b.ndim not in (1, 2):
        raise ValueError(f"b must be 1D or 2D, got shape {b.shape}")
    if Ai.ndim != 1 or Ai.shape != Aj.shape or Ax.shape != Ai.shape:
        raise ValueError(
            f"Ai, Aj, Ax must be 1D with equal lengths, got "
            f"{Ai.shape}, {Aj.shape}, {Ax.shape}"
        )

    if mode != MODE_A:
        # Factor-part solves are building blocks (e.g. MVN sampling); AD
        # through them is not defined.
        return _solve_ffi(Ai, Aj, Ax, b, mode)

    # AD: for x = A^{-1} b with A symmetric, v = A^{-1} g gives db = v and,
    # for the stored upper-triangle entry (i, j) which appears in the matrix
    # at both (i, j) and (j, i), dAx = -(v_i x_j + v_j x_i) (i < j) or
    # -v_i x_i (i == j). Ignored lower-triangle entries get zero.
    @jax.custom_vjp
    def _solve_a(Ax, b):
        return _solve_ffi(Ai, Aj, Ax, b, MODE_A)

    def _fwd(Ax, b):
        x = _solve_ffi(Ai, Aj, Ax, b, MODE_A)
        return x, (Ax, x)

    def _bwd(res, g):
        Ax_saved, x = res
        v = _solve_ffi(Ai, Aj, Ax_saved, g, MODE_A)
        if x.ndim == 1:
            cross = v[Ai] * x[Aj] + v[Aj] * x[Ai]
            diag = v[Ai] * x[Ai]
        else:
            cross = (v[Ai] * x[Aj]).sum(-1) + (v[Aj] * x[Ai]).sum(-1)
            diag = (v[Ai] * x[Ai]).sum(-1)
        dAx = -jnp.where(Ai == Aj, diag, jnp.where(Ai < Aj, cross, 0.0))
        return dAx, v

    _solve_a.defvjp(_fwd, _bwd)
    return _solve_a(Ax, b)


def logdet(Ai, Aj, Ax, n):
    """Log-determinant of a symmetric positive definite sparse matrix.

    Computed from the Cholesky factor's diagonal, sharing the factorization
    cache with :func:`solve`: a ``solve`` and a ``logdet`` with identical
    values factorize only once.

    Args:
        Ai, Aj, Ax: COO matrix as in :func:`solve`.
        n: matrix dimension (static Python int).

    Returns:
        Scalar float64 ``log(det(A))``. Not differentiable.
    """
    _require_x64()
    Ai = jnp.asarray(Ai, jnp.int32)
    Aj = jnp.asarray(Aj, jnp.int32)
    Ax = jnp.asarray(Ax, jnp.float64)
    call = jax.ffi.ffi_call(
        "cholmodjax_logdet_f64",
        jax.ShapeDtypeStruct((), jnp.float64),
        vmap_method="sequential",
    )
    return call(Ai, Aj, Ax, n=np.int64(n))


def update_solve(Ai, Aj, Ax, C, b, downdate=False, mode=MODE_A, return_logdet=False):
    """Solve ``(A ± C C') x = b`` via a rank-k update of ``A``'s factor.

    ``A`` (the COO matrix) is factored once and cached; each call rebuilds a
    working copy of that factor, applies CHOLMOD's ``cholmod_updown`` for the
    low-rank term ``C C'``, and solves. When ``A`` is held fixed and only the
    low-rank term ``C`` varies, this is much cheaper than assembling and
    refactoring ``A ± C C'`` from scratch (``O(k·nnz(L))`` vs. a full
    factorization). The cached factor of ``A`` is never mutated.

    Args:
        Ai, Aj, Ax: COO of the symmetric positive definite base matrix ``A``,
            as in :func:`solve`.
        C: ``[n]`` or ``[n, k]`` float64 — the rank-k update columns. A 1D
            ``C`` is treated as a single column (rank-1). May be sparse in
            content (explicit zeros are dropped before the update).
        b: ``[n]`` or ``[n, n_rhs]`` float64 — right-hand side(s).
        downdate: if ``True`` solve ``(A - C C') x = b`` (downdate); otherwise
            ``(A + C C') x = b`` (update).
        mode: solve mode, as in :func:`solve`.
        return_logdet: if ``True`` also return ``log|A ± C C'|`` (computed from
            the same updated factor, so it is essentially free).

    Returns:
        ``x`` with the same shape as ``b``, or ``(x, logdet)`` if
        ``return_logdet`` is set.

    Raises:
        Exception: if ``A`` is not positive definite, or a downdate drives the
            updated matrix indefinite.
    """
    _require_x64()
    Ai = jnp.asarray(Ai, jnp.int32)
    Aj = jnp.asarray(Aj, jnp.int32)
    Ax = jnp.asarray(Ax, jnp.float64)
    C = jnp.asarray(C, jnp.float64)
    b = jnp.asarray(b, jnp.float64)
    if C.ndim == 1:
        C = C[:, None]
    if C.ndim != 2:
        raise ValueError(f"C must be 1D or 2D (n, k), got shape {C.shape}")
    if b.ndim not in (1, 2):
        raise ValueError(f"b must be 1D or 2D, got shape {b.shape}")
    if Ai.ndim != 1 or Ai.shape != Aj.shape or Ax.shape != Ai.shape:
        raise ValueError("Ai, Aj, Ax must be 1D with equal lengths")

    call = jax.ffi.ffi_call(
        "cholmodjax_updown_solve_f64",
        (
            jax.ShapeDtypeStruct(b.shape, b.dtype),
            jax.ShapeDtypeStruct((), jnp.float64),
        ),
        vmap_method="sequential",
    )
    x, ld = call(
        Ai,
        Aj,
        Ax,
        C,
        b,
        mode=np.int64(mode),
        downdate=np.int64(1 if downdate else 0),
    )
    return (x, ld) if return_logdet else x


def _bcoo_parts(A):
    """Extract ``(Ai, Aj, Ax)`` from a JAX ``BCOO``-like sparse matrix.

    Duck-typed on ``.indices`` / ``.data`` so there is no hard dependency on
    ``jax.experimental.sparse``. Only a plain 2D matrix (no batch or dense
    dimensions) is supported.
    """
    idx = getattr(A, "indices", None)
    data = getattr(A, "data", None)
    if idx is None or data is None:
        raise TypeError(
            "cholmodjax: expected a BCOO-like matrix with .indices and .data"
        )
    if getattr(A, "n_batch", 0) or getattr(A, "n_dense", 0):
        raise ValueError(
            "cholmodjax: only a plain 2D BCOO (n_batch=0, n_dense=0) is supported"
        )
    if idx.ndim != 2 or idx.shape[-1] != 2:
        raise ValueError(
            f"cholmodjax: expected BCOO indices of shape (nnz, 2), got {idx.shape}"
        )
    return idx[:, 0], idx[:, 1], data


def solve_bcoo(A, b, mode=MODE_A):
    """:func:`solve` for a JAX ``BCOO`` matrix ``A``.

    Equivalent to ``solve(A.indices[:, 0], A.indices[:, 1], A.data, b, ...)``.
    A full-symmetric BCOO works directly — only the upper triangle is read —
    and duplicate/unsorted entries are handled. The pattern cache keys on the
    index values, so the analysis-reuse speedup applies whenever ``A``'s
    sparsity pattern is stable across calls.
    """
    Ai, Aj, Ax = _bcoo_parts(A)
    return solve(Ai, Aj, Ax, b, mode=mode)


def logdet_bcoo(A):
    """:func:`logdet` for a JAX ``BCOO`` matrix ``A``."""
    Ai, Aj, Ax = _bcoo_parts(A)
    return logdet(Ai, Aj, Ax, A.shape[0])


def update_solve_bcoo(A, C, b, downdate=False, mode=MODE_A, return_logdet=False):
    """:func:`update_solve` for a JAX ``BCOO`` base matrix ``A``."""
    Ai, Aj, Ax = _bcoo_parts(A)
    return update_solve(
        Ai,
        Aj,
        Ax,
        C,
        b,
        downdate=downdate,
        mode=mode,
        return_logdet=return_logdet,
    )


def set_options(*, supernodal="auto"):
    """Configure CHOLMOD behavior.

    Args:
        supernodal: ``"auto"`` (CHOLMOD decides, default), ``"simplicial"``
            (faster triangular solves on very sparse matrices), or
            ``"supernodal"`` (BLAS-based, faster factorization on denser
            problems). Clears the factorization cache so the setting applies
            to all patterns.
    """
    codes = {"simplicial": 0, "auto": 1, "supernodal": 2}
    if supernodal not in codes:
        raise ValueError(f"supernodal must be one of {sorted(codes)}")
    _cpp.set_supernodal(codes[supernodal])
    _cpp.clear_cache()


def clear_cache():
    """Free all cached symbolic analyses and factorizations."""
    _cpp.clear_cache()


def cache_size():
    """Number of sparsity patterns currently cached."""
    return _cpp.cache_size()
