"""cholgraph: JAX-native sparse Cholesky via CHOLMOD.

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
    import cholgraph

    jax.config.update("jax_enable_x64", True)

    @jax.jit
    def step(Ax, b):
        return cholgraph.solve(Ai, Aj, Ax, b)   # full CHOLMOD speed in JIT

``solve`` is differentiable in ``Ax`` and ``b``, and ``logdet`` is
differentiable in ``Ax`` (reverse mode) via the selected inverse (:func:`selinv`)
— together the two pieces of a Gaussian log-density's gradient. Under
``jax.vmap`` the whole batch is solved in a single native FFI call that loops
in C++ and reuses the cached analysis. :func:`update_solve` exposes CHOLMOD's
rank-k update/downdate for cheap solves of ``(A ± C C') x = b`` with ``A`` held
fixed.
"""

import jax
import jax.numpy as jnp
import numpy as np

import cholgraph_cpp as _cpp

__version__ = "0.6.0"
__all__ = [
    "solve",
    "logdet",
    "selinv",
    "factor_solve",
    "sample_gaussian",
    "update_solve",
    "solve_bcoo",
    "logdet_bcoo",
    "update_solve_bcoo",
    "clear_cache",
    "cache_size",
    "factorization_count",
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
    "cholgraph_solve_f64", _cpp.solve_f64_capsule(), platform="cpu"
)
jax.ffi.register_ffi_target(
    "cholgraph_solve_batched_f64",
    _cpp.solve_batched_f64_capsule(),
    platform="cpu",
)
jax.ffi.register_ffi_target(
    "cholgraph_logdet_f64", _cpp.logdet_f64_capsule(), platform="cpu"
)
jax.ffi.register_ffi_target(
    "cholgraph_selinv_f64", _cpp.selinv_f64_capsule(), platform="cpu"
)
jax.ffi.register_ffi_target(
    "cholgraph_updown_solve_f64",
    _cpp.updown_solve_f64_capsule(),
    platform="cpu",
)
jax.ffi.register_ffi_target(
    "cholgraph_factor_solve_f64",
    _cpp.factor_solve_f64_capsule(),
    platform="cpu",
)
jax.ffi.register_ffi_target(
    "cholgraph_factor_solve_batched_f64",
    _cpp.factor_solve_batched_f64_capsule(),
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
            "cholgraph requires 64-bit mode. Call "
            'jax.config.update("jax_enable_x64", True) before using it.'
        )


def _solve_batched(Ai, Aj, Ax, b, mode):
    # One FFI call for a whole batch (leading axis 0): Ax is (B, nnz), b is
    # (B, n[, nrhs]); the C++ handler loops over B reusing the cached analysis.
    call = jax.ffi.ffi_call(
        "cholgraph_solve_batched_f64", jax.ShapeDtypeStruct(b.shape, b.dtype)
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
            "cholgraph_solve_f64", jax.ShapeDtypeStruct(b.shape, b.dtype)
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


def _logdet_ffi(Ai, Aj, Ax, n):
    call = jax.ffi.ffi_call(
        "cholgraph_logdet_f64",
        jax.ShapeDtypeStruct((), jnp.float64),
        vmap_method="sequential",
    )
    return call(Ai, Aj, Ax, n=np.int64(n))


def _selinv_ffi(Ai, Aj, Ax, n):
    call = jax.ffi.ffi_call(
        "cholgraph_selinv_f64",
        jax.ShapeDtypeStruct(Ax.shape, jnp.float64),
        vmap_method="sequential",
    )
    return call(Ai, Aj, Ax, n=np.int64(n))


def selinv(Ai, Aj, Ax, n):
    """Selected inverse: entries of ``A^{-1}`` at ``A``'s sparsity pattern.

    Returns ``z`` with ``z[k] == (A^{-1})[Ai[k], Aj[k]]``, computed by
    Takahashi's recurrence on the Cholesky factor without ever forming the dense
    inverse. Shares the factorization cache with :func:`solve` / :func:`logdet`.

    This is the quantity :func:`logdet`'s reverse-mode rule uses
    (``d log|A| / dA = A^{-1}``), exposed directly because the selected inverse
    is useful on its own — e.g. posterior marginal variances ``diag(A^{-1})`` of
    a Gaussian with precision ``A``, read off the ``Ai == Aj`` entries.

    Args:
        Ai, Aj, Ax: COO matrix as in :func:`solve`.
        n: matrix dimension (static Python int).

    Returns:
        ``[n_nz]`` float64, aligned with ``Ax``: ``A^{-1}`` at each COO position.
        Off-diagonal values are symmetric (``(A^{-1})[i, j] == (A^{-1})[j, i]``),
        so entries below the diagonal carry the same value as their transpose.
    """
    _require_x64()
    Ai = jnp.asarray(Ai, jnp.int32)
    Aj = jnp.asarray(Aj, jnp.int32)
    Ax = jnp.asarray(Ax, jnp.float64)
    if Ai.ndim != 1 or Ai.shape != Aj.shape or Ax.shape != Ai.shape:
        raise ValueError("Ai, Aj, Ax must be 1D with equal lengths")
    return _selinv_ffi(Ai, Aj, Ax, int(n))


def logdet(Ai, Aj, Ax, n):
    """Log-determinant of a symmetric positive definite sparse matrix.

    Computed from the Cholesky factor's diagonal, sharing the factorization
    cache with :func:`solve`: a ``solve`` and a ``logdet`` with identical
    values factorize only once.

    Differentiable in ``Ax`` (reverse mode). Since ``d log|A| / dA = A^{-1}``,
    the gradient is the :func:`selinv` selected inverse at the COO positions
    (with off-diagonal entries doubled for the symmetric pair), so it costs one
    selected-inversion pass over the factor — no dense inverse. This makes
    ``logdet`` usable inside a JAX log-density for gradient-based inference
    (e.g. HMC/NUTS via numpyro/blackjax, or empirical-Bayes optimization of the
    matrix values), pairing with :func:`solve`'s VJP for the quadratic-form term.

    Args:
        Ai, Aj, Ax: COO matrix as in :func:`solve`.
        n: matrix dimension (static Python int).

    Returns:
        Scalar float64 ``log(det(A))``.
    """
    _require_x64()
    Ai = jnp.asarray(Ai, jnp.int32)
    Aj = jnp.asarray(Aj, jnp.int32)
    Ax = jnp.asarray(Ax, jnp.float64)
    n = int(n)

    # AD: log|A| depends on Ax only through A. d log|A| = tr(A^{-1} dA), and for
    # a stored upper-triangle value at (i, j) that sets both A_ij and A_ji the
    # sensitivity is (A^{-1})_ij + (A^{-1})_ji = 2 (A^{-1})_ij; a diagonal value
    # contributes (A^{-1})_ii once; ignored lower-triangle entries get zero.
    @jax.custom_vjp
    def _logdet_a(Ax):
        return _logdet_ffi(Ai, Aj, Ax, n)

    def _fwd(Ax):
        return _logdet_ffi(Ai, Aj, Ax, n), Ax

    def _bwd(Ax_saved, g):
        z = _selinv_ffi(Ai, Aj, Ax_saved, n)
        dAx = g * jnp.where(Ai == Aj, z, jnp.where(Ai < Aj, 2.0 * z, 0.0))
        return (dAx,)

    _logdet_a.defvjp(_fwd, _bwd)
    return _logdet_a(Ax)


# factor_solve dispatchers, cached by (mode_chain, chain_lens, want_logdet, n).
_FS_DISPATCH = {}


def _make_factor_solve_dispatch(mode_chain, chain_lens, want_logdet, n):
    mc = np.asarray(mode_chain, np.int64)
    cl = np.asarray(chain_lens, np.int64)
    wl = np.int64(1 if want_logdet else 0)
    nn = np.int64(n)

    def _out_shapes(bs, batch=None):
        shapes = [jax.ShapeDtypeStruct(b.shape, b.dtype) for b in bs]
        if want_logdet:
            ld_shape = () if batch is None else (batch,)
            shapes.append(jax.ShapeDtypeStruct(ld_shape, jnp.float64))
        return shapes

    def _unbatched(Ai, Aj, Ax, bs):
        call = jax.ffi.ffi_call("cholgraph_factor_solve_f64", _out_shapes(bs))
        return tuple(
            call(Ai, Aj, Ax, *bs, mode_chain=mc, chain_lens=cl,
                 want_logdet=wl, n=nn)
        )

    dispatch = jax.custom_batching.custom_vmap(_unbatched)

    @dispatch.def_vmap
    def _rule(axis_size, in_batched, Ai, Aj, Ax, bs):
        Ai_b, Aj_b, Ax_b, bs_b = in_batched
        if Ai_b or Aj_b:
            raise ValueError(
                "cholgraph.factor_solve: cannot vmap over the sparsity "
                "pattern (Ai, Aj) — it must be shared across the batch"
            )
        if not Ax_b:
            Ax = jnp.broadcast_to(Ax, (axis_size,) + Ax.shape)
        bs = tuple(
            b if bb else jnp.broadcast_to(b, (axis_size,) + b.shape)
            for b, bb in zip(bs, bs_b)
        )
        call = jax.ffi.ffi_call(
            "cholgraph_factor_solve_batched_f64", _out_shapes(bs, axis_size)
        )
        outs = tuple(
            call(Ai, Aj, Ax, *bs, mode_chain=mc, chain_lens=cl,
                 want_logdet=wl, n=nn)
        )
        return outs, (True,) * len(outs)

    return dispatch


def factor_solve(Ai, Aj, Ax, rhs, want_logdet=False, n=None):
    """Factor ``A`` once, then run several solve *chains* from that one factor.

    This is the "factor once, do everything" primitive: ``A`` (the COO matrix)
    is factored a single time and every requested solve — plus an optional
    ``logdet`` — is served from that factor. Under ``jax.vmap`` it lowers to one
    batched FFI call that factors **once per batch element** regardless of how
    many chains are requested, which is the key win for Gibbs sweeps that
    otherwise pay for the same factorization several times (mean, sample, and
    density all need it).

    Each entry of ``rhs`` is ``(b, modes)`` where ``modes`` is a single
    ``MODE_*`` code or a sequence of them applied left to right — a chain. For a
    chain ``[m0, m1, ...]`` the result is ``... m1(m0(b))``. Examples:

    - posterior mean ``A^{-1} b``: ``(b, MODE_A)``
    - a draw's factor part ``P' L^{-T} z``: ``(z, [MODE_LT, MODE_PT])``

    Args:
        Ai, Aj, Ax: COO of the SPD matrix ``A``, as in :func:`solve`.
        rhs: list of ``(b, modes)``. Each ``b`` is ``[n]`` or ``[n, n_rhs]``.
        want_logdet: if ``True``, also return ``log|A|`` from the same factor.
        n: matrix dimension. Inferred from ``rhs[0]`` when omitted; required if
            ``rhs`` is empty (logdet only).

    Returns:
        A list ``xs`` of solutions, one per ``rhs`` entry, each matching its
        ``b``'s shape. If ``want_logdet`` is set, returns ``(xs, logdet)``.

    Note:
        This primitive is forward-only (no autodiff rule). Use :func:`solve` /
        :func:`logdet`, which define VJPs, when you need gradients.
    """
    _require_x64()
    Ai = jnp.asarray(Ai, jnp.int32)
    Aj = jnp.asarray(Aj, jnp.int32)
    Ax = jnp.asarray(Ax, jnp.float64)
    if Ai.ndim != 1 or Ai.shape != Aj.shape or Ax.shape != Ai.shape:
        raise ValueError("Ai, Aj, Ax must be 1D with equal lengths")

    bs, chain_lens, mode_chain = [], [], []
    for entry in rhs:
        b, modes = entry
        b = jnp.asarray(b, jnp.float64)
        if b.ndim not in (1, 2):
            raise ValueError(f"each rhs must be 1D or 2D, got shape {b.shape}")
        if isinstance(modes, (int, np.integer)):
            modes = (int(modes),)
        else:
            modes = tuple(int(m) for m in modes)
        if any(m < 0 or m > 8 for m in modes):
            raise ValueError(f"invalid solve mode in chain {modes}")
        bs.append(b)
        chain_lens.append(len(modes))
        mode_chain.extend(modes)

    if n is None:
        if not bs:
            raise ValueError("n must be given when rhs is empty")
        n = int(bs[0].shape[0])
    n = int(n)

    key = (tuple(mode_chain), tuple(chain_lens), bool(want_logdet), n)
    dispatch = _FS_DISPATCH.get(key)
    if dispatch is None:
        dispatch = _FS_DISPATCH[key] = _make_factor_solve_dispatch(*key)

    outs = list(dispatch(Ai, Aj, Ax, tuple(bs)))
    if want_logdet:
        return outs[:-1], outs[-1]
    return outs


def sample_gaussian(Ai, Aj, Ax, b, z, want_logdet=False):
    """Draw ``eta ~ N(A^{-1} b, A^{-1})`` from a single factorization of ``A``.

    The Gibbs Gaussian-conditional step in one factorization: the posterior
    mean ``A^{-1} b`` and the correlated draw share one Cholesky factor. With
    ``A = P' L L' P``, a sample is ``eta = mean + P' L^{-T} z`` for
    ``z ~ N(0, I)``, since ``Cov(P' L^{-T} z) = A^{-1}``. Built on
    :func:`factor_solve`, so under ``vmap`` it factors once per element.

    Args:
        Ai, Aj, Ax: COO of the SPD precision matrix ``A``.
        b: ``[n]`` or ``[n, n_rhs]`` — mean numerator (posterior mean is
            ``A^{-1} b``).
        z: standard-normal draw with the same shape as ``b``.
        want_logdet: if ``True``, also return ``log|A|``.

    Returns:
        ``(eta, mean)``, or ``(eta, mean, logdet)`` if ``want_logdet``.
    """
    xs = factor_solve(
        Ai, Aj, Ax,
        [(b, MODE_A), (z, (MODE_LT, MODE_PT))],
        want_logdet=want_logdet,
    )
    if want_logdet:
        (mean, w), ld = xs
        return mean + w, mean, ld
    mean, w = xs
    return mean + w, mean


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
        "cholgraph_updown_solve_f64",
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
            "cholgraph: expected a BCOO-like matrix with .indices and .data"
        )
    if getattr(A, "n_batch", 0) or getattr(A, "n_dense", 0):
        raise ValueError(
            "cholgraph: only a plain 2D BCOO (n_batch=0, n_dense=0) is supported"
        )
    if idx.ndim != 2 or idx.shape[-1] != 2:
        raise ValueError(
            f"cholgraph: expected BCOO indices of shape (nnz, 2), got {idx.shape}"
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


def factorization_count():
    """Total numeric (re)factorizations performed since import.

    Bumped only on a real factorization, not on the value-cache skip. Useful
    for confirming that a fused :func:`factor_solve` / :func:`sample_gaussian`
    sweep factors ``A`` once rather than once per solve.
    """
    return _cpp.factorization_count()
