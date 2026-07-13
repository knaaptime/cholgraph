# cholmodjax

JAX-native sparse Cholesky via [CHOLMOD](https://github.com/DrTimothyAldenDavis/SuiteSparse).
Solves with symmetric positive definite sparse matrices run at full native speed
**inside `@jax.jit`** (and `lax.scan` / `lax.fori_loop`) — no Python callback overhead.

## Why?

- No open-source JIT framework (JAX, PyTorch, TensorFlow) exposes sparse Cholesky as a
  compilable primitive; [`klujax`](https://github.com/florislaporte/klujax) covers sparse LU,
  which is ~2× slower than Cholesky for SPD systems.
- The target workload is Gibbs samplers (e.g. Bayesian spatial econometrics), where an SPD
  precision matrix is solved thousands of times with the same sparsity pattern but changing
  values, inside a JIT-compiled loop.
- Benchmarks (M-series macOS, 2D grid Laplacian, values changing every iteration): matches a
  hand-written scikit-sparse Python loop per iteration and is ~2.7× faster than
  `scipy.sparse.linalg.splu`, while running entirely inside `jax.jit`.

## How it works

`solve` and `logdet` are [XLA FFI](https://docs.jax.dev/en/latest/ffi.html) custom calls into
CHOLMOD. The extension caches symbolic analyses (fill-reducing ordering + elimination tree)
keyed on the sparsity pattern, so repeated calls with the same pattern only pay for the numeric
refactorization — and calls with unchanged values skip even that, sharing one factorization
between `solve` and `logdet`. There are no handles to manage and nothing to pass through JIT
boundaries; the caching is transparent.

## Installation

```bash
conda env create -f environment.yml   # suitesparse, jax, nanobind, cmake, ...
conda activate cholmodjax
pip install --no-build-isolation .
```

## Quick start

```python
import jax
import jax.numpy as jnp
import numpy as np
import cholmodjax

jax.config.update("jax_enable_x64", True)   # required: CHOLMOD is float64

# SPD matrix in COO form. Entries with Ai <= Aj are used (upper triangle);
# pass the full symmetric matrix or just its upper triangle.
Ai = np.array([0, 0, 1, 1, 1, 2, 2], dtype=np.int32)
Aj = np.array([0, 1, 0, 1, 2, 1, 2], dtype=np.int32)
Ax = jnp.array([4.0, 1.0, 1.0, 5.0, 2.0, 2.0, 6.0])
b = jnp.array([1.0, 2.0, 3.0])

x = cholmodjax.solve(Ai, Aj, Ax, b)          # eager
ld = cholmodjax.logdet(Ai, Aj, Ax, n=3)      # log|A| from the same factorization

@jax.jit                                      # ...or fully JIT-compiled
def gibbs_step(Ax, b):
    x = cholmodjax.solve(Ai, Aj, Ax, b)      # full CHOLMOD speed, no callbacks
    ld = cholmodjax.logdet(Ai, Aj, Ax, n=3)  # factorization shared with solve
    return x, ld
```

Features:

- **`jit` / `lax.scan`**: the symbolic analysis is computed once and reused across iterations.
- **Autodiff**: `solve` has a custom VJP (reverse-mode) in both `Ax` and `b`.
- **`vmap`**: `jax.vmap(solve)` lowers to a *single* native FFI call that loops over the batch
  in C++ (reusing the cached analysis), rather than XLA per-iteration dispatch. Composes with
  `grad` (`vmap(grad(solve))` batches too). Map over `Ax`, `b`, or both.
- **Multiple right-hand sides**: `b` may be `(n,)` or `(n, n_rhs)`.
- **Factor-part solves**: `mode=cholmodjax.MODE_LT` etc. expose CHOLMOD's solve systems
  (`P' L L' P = A`). Sampling `y ~ N(0, A^{-1})`:
  `y = solve(..., solve(..., z, mode=MODE_LT), mode=MODE_PT)`.
- **Not positive definite** → runtime exception (the factor is always a true LL').

## Factor once, do everything: `factor_solve` / `sample_gaussian`

`solve` and `logdet` are separate primitives, so a Gibbs sweep that needs a posterior mean, a
correlated draw, and a log-determinant factors the *same* A several times — and under `vmap`
that is one factorization per solve **per batch element**. `factor_solve` factors A once and
serves every requested solve (each a *chain* of `MODE_*` codes) plus an optional `logdet` from
that single factor. Under `vmap` it lowers to one batched FFI call that factors **once per
element**, whatever the number of chains.

```python
# Gibbs Gaussian step: posterior mean, a draw ~ N(mean, A^-1), and log|A|,
# from ONE factorization. eta = mean + P' L^-T z  (since A = P' L L' P).
eta, mean, ld = cholmodjax.sample_gaussian(Ai, Aj, Ax, b, z, want_logdet=True)

# ...or spell it out with the general primitive:
(mean, w), ld = cholmodjax.factor_solve(
    Ai, Aj, Ax,
    [(b, cholmodjax.MODE_A),                          # A^-1 b
     (z, (cholmodjax.MODE_LT, cholmodjax.MODE_PT))],  # P' L^-T z  (chain)
    want_logdet=True)
eta = mean + w
```

Each `rhs` entry is `(b, modes)` where `modes` is one `MODE_*` or a sequence applied left to
right. `cholmodjax.factorization_count()` reports how many real factorizations have happened —
handy for confirming the fusion. Benchmarked Gibbs draw (mean + sample + logdet) vmapped over a
batch of *different* A's: **4× fewer factorizations and ~3.3–3.6× faster** than issuing the
separate `solve`/`logdet` primitives. `factor_solve` is forward-only (no autodiff rule); use
`solve`/`logdet` when you need gradients.

## JAX sparse (`BCOO`)

JAX's native sparse type is `jax.experimental.sparse.BCOO`, whose `.indices` is `(nnz, 2)` and
`.data` is `(nnz,)`. Convenience wrappers accept one directly:

```python
from jax.experimental import sparse as jsparse
A = jsparse.BCOO.fromdense(A_dense)         # or build however you like

x  = cholmodjax.solve_bcoo(A, b)            # == solve(A.indices[:,0], A.indices[:,1], A.data, b)
ld = cholmodjax.logdet_bcoo(A)
x  = cholmodjax.update_solve_bcoo(A, c, b)  # rank-k update, as below
```

The analysis-reuse speedup is unaffected: the pattern cache keys on the concrete index values
(exactly a BCOO's `.indices`), so a stable pattern across `jit`/`vmap` calls keeps hitting the
cache. A full-symmetric BCOO works directly (only the upper triangle is read), and
unsorted/duplicate entries are handled. Only a plain 2D BCOO (`n_batch=0`, `n_dense=0`) is
supported.

## Rank-k update / downdate

`update_solve` solves `(A ± C C') x = b` by applying CHOLMOD's `cholmod_updown` to a working
copy of `A`'s cached factor, instead of refactoring the modified matrix from scratch. `A` is
factored once; each call is `O(k · path)` where `path` is the elimination-tree path touched by
`C`'s nonzeros.

```python
# Add an observation (rank-1, sparse update column) and re-solve, cheaply:
x = cholmodjax.update_solve(Ai, Aj, Ax, c, b)                 # (A + c c') x = b
x = cholmodjax.update_solve(Ai, Aj, Ax, c, b, downdate=True)  # (A - c c') x = b
x, ld = cholmodjax.update_solve(Ai, Aj, Ax, C, b, return_logdet=True)  # C is (n, k)
```

**When it pays off:** the update column(s) `C` must be **sparse** (a few nonzeros — e.g. one
data point and its neighbors). On the grid-Laplacian benchmark a rank-1 sparse update is ~3×
faster than a full factorize+solve. A *dense* `C` walks the whole tree and is slower than
refactoring — use plain `solve` on the reassembled matrix in that case. The base cached factor
is never mutated, so `update_solve` is a pure function (works under `jit`; not differentiable).

## Options

```python
cholmodjax.set_options(supernodal="simplicial")  # or "auto" (default), "supernodal"
cholmodjax.clear_cache()                         # free cached factorizations
```

For very sparse matrices (e.g. planar/grid graphs), `"simplicial"` often gives faster
triangular solves; `"supernodal"` (BLAS-based) wins on denser problems. `"auto"` lets
CHOLMOD choose based on the matrix.

## Status / roadmap

- [x] `solve` (all CHOLMOD solve modes), `logdet`, symbolic + numeric caching, custom VJP,
      multi-RHS, tests, benchmarks
- [x] Native batching: `jax.vmap(solve)` → one FFI call looping over the batch in C++
- [x] `cholmod_updown` rank-k update/downdate (`update_solve`)
- [x] Cache the simplicial LDL' base factor for updown (rebuilt only on refactor), so the
      LL'→LDL' conversion is paid once per base change, not once per call
- [x] `factor_solve` / `sample_gaussian`: factor once, serve many solve chains + logdet from
      one factor; fuses under `vmap` to one factorization per batch element
- [ ] float32 (CHOLMOD 5 single precision) and int64 indices
- [ ] Autodiff rule for `factor_solve` (currently forward-only)
- [ ] Wheels / conda-forge packaging

## License

MIT
