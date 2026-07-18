// cholmodjax: CHOLMOD sparse Cholesky as XLA FFI custom calls.
//
// Design:
//   - A single global cholmod_common guarded by a mutex (CHOLMOD's workspace
//     is not thread-safe; BLAS parallelism inside CHOLMOD is unaffected).
//   - Symbolic analyses are cached in a registry keyed by a hash of the
//     sparsity pattern (n, Ai, Aj), with full memcmp verification on hit.
//     Repeated solves with the same pattern reuse cholmod_analyze() output;
//     repeated solves with identical values also skip cholmod_factorize().
//   - Input is COO of a symmetric matrix. Entries with i <= j (upper triangle
//     plus diagonal) are used; entries with i > j are ignored, so both
//     "full symmetric" and "upper-only" input work. Duplicates are summed.
//
// The FFI handlers never touch Python and hold no GIL, so they run at full
// native speed inside XLA-compiled programs.

#include <cholmod.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
namespace nb = nanobind;

// ---------------------------------------------------------------------------
// Global CHOLMOD state
// ---------------------------------------------------------------------------

static std::mutex g_mutex;
static cholmod_common g_common;
static bool g_started = false;

static void error_handler(int status, const char* file, int line,
                          const char* message) {
  // Errors are surfaced as XLA errors; keep CHOLMOD off stderr.
  (void)status;
  (void)file;
  (void)line;
  (void)message;
}

static void ensure_started_locked() {
  if (!g_started) {
    cholmod_start(&g_common);
    g_common.print = 0;
    g_common.error_handler = error_handler;
    // Always produce an LL' factor: simplicial LDL' would silently accept
    // indefinite matrices, and an actual L is needed for MODE_L / MODE_LT
    // solves (e.g. sampling from N(0, A^{-1})).
    g_common.final_ll = 1;
    g_started = true;
  }
}

// ---------------------------------------------------------------------------
// Pattern registry (symbolic analysis cache)
// ---------------------------------------------------------------------------

struct PatternEntry {
  int64_t n = 0;
  std::vector<int32_t> Ai, Aj;   // full COO copy, for exact hit verification
  std::vector<int64_t> pos;      // COO k -> offset into A->x, -1 if i > j
  cholmod_sparse* A = nullptr;   // upper-triangular CSC, stype = 1
  cholmod_factor* L = nullptr;
  std::vector<double> last_Ax;   // values of the last successful factorization
  double logdet = 0.0;           // log|A| of the last successful factorization
  bool factored = false;
  // Persistent cholmod_solve2 workspaces, reused across solves.
  cholmod_dense* Xwork = nullptr;
  cholmod_dense* Ywork = nullptr;
  cholmod_dense* Ework = nullptr;
  // updown scratch factors. Lldl is a persistent simplicial LDL' copy of the
  // base factor, rebuilt only when the base is refactored (tracked by epoch);
  // Lupd is a per-call working copy of Lldl that cholmod_updown mutates, so the
  // base factor is never touched.
  cholmod_factor* Lldl = nullptr;
  cholmod_factor* Lupd = nullptr;
  int64_t factor_epoch = 0;   // bumped on each real (re)factorization of L
  int64_t ldl_epoch = -1;     // epoch Lldl was built from; -1 == not built
};

static std::unordered_map<uint64_t, std::vector<std::unique_ptr<PatternEntry>>>
    g_registry;
// Fast path: solver loops hit the same pattern every call.
static PatternEntry* g_last_entry = nullptr;
// Count of actual numeric (re)factorizations, for tests/introspection. Bumped
// only when factorize_locked truly refactors (not on the value-cache skip).
static int64_t g_num_factorizations = 0;

static uint64_t fnv1a(const void* data, size_t nbytes, uint64_t h) {
  const unsigned char* p = static_cast<const unsigned char*>(data);
  size_t nwords = nbytes / 8;
  for (size_t i = 0; i < nwords; ++i) {
    uint64_t v;
    std::memcpy(&v, p + i * 8, 8);
    h ^= v;
    h *= 1099511628211ULL;
  }
  for (size_t i = nwords * 8; i < nbytes; ++i) {
    h ^= p[i];
    h *= 1099511628211ULL;
  }
  return h;
}

static uint64_t pattern_hash(const int32_t* Ai, const int32_t* Aj, int64_t nnz,
                             int64_t n) {
  uint64_t h = 14695981039346656037ULL;
  h = fnv1a(&n, sizeof(n), h);
  h = fnv1a(Ai, nnz * sizeof(int32_t), h);
  h = fnv1a(Aj, nnz * sizeof(int32_t), h);
  return h;
}

static void free_entry_locked(PatternEntry* e) {
  if (e->L) cholmod_free_factor(&e->L, &g_common);
  if (e->A) cholmod_free_sparse(&e->A, &g_common);
  if (e->Xwork) cholmod_free_dense(&e->Xwork, &g_common);
  if (e->Ywork) cholmod_free_dense(&e->Ywork, &g_common);
  if (e->Ework) cholmod_free_dense(&e->Ework, &g_common);
  if (e->Lldl) cholmod_free_factor(&e->Lldl, &g_common);
  if (e->Lupd) cholmod_free_factor(&e->Lupd, &g_common);
}

// Build CSC (upper triangle, sorted, duplicates merged) and run
// cholmod_analyze. Returns nullptr and sets `err` on failure.
static PatternEntry* create_entry_locked(const int32_t* Ai, const int32_t* Aj,
                                         int64_t nnz, int64_t n,
                                         std::string* err) {
  for (int64_t k = 0; k < nnz; ++k) {
    if (Ai[k] < 0 || Ai[k] >= n || Aj[k] < 0 || Aj[k] >= n) {
      *err = "cholmodjax: COO index out of range for matrix dimension " +
             std::to_string(n);
      return nullptr;
    }
  }

  auto entry = std::make_unique<PatternEntry>();
  entry->n = n;
  entry->Ai.assign(Ai, Ai + nnz);
  entry->Aj.assign(Aj, Aj + nnz);
  entry->pos.assign(nnz, -1);

  // Upper-triangle entries sorted by (col, row); keep original k for mapping.
  struct Trip {
    int32_t i, j;
    int64_t k;
  };
  std::vector<Trip> upper;
  upper.reserve(nnz);
  for (int64_t k = 0; k < nnz; ++k)
    if (Ai[k] <= Aj[k]) upper.push_back({Ai[k], Aj[k], k});
  std::sort(upper.begin(), upper.end(), [](const Trip& a, const Trip& b) {
    return a.j != b.j ? a.j < b.j : a.i < b.i;
  });

  // Count unique (i, j) slots.
  int64_t nnz_csc = 0;
  for (size_t t = 0; t < upper.size(); ++t)
    if (t == 0 || upper[t].i != upper[t - 1].i || upper[t].j != upper[t - 1].j)
      ++nnz_csc;

  cholmod_sparse* A = cholmod_allocate_sparse(
      n, n, nnz_csc, /*sorted=*/1, /*packed=*/1, /*stype=*/1, CHOLMOD_REAL,
      &g_common);
  if (!A) {
    *err = "cholmodjax: cholmod_allocate_sparse failed";
    return nullptr;
  }

  int32_t* Ap = static_cast<int32_t*>(A->p);
  int32_t* Ar = static_cast<int32_t*>(A->i);
  double* Axv = static_cast<double*>(A->x);
  std::memset(Ap, 0, (n + 1) * sizeof(int32_t));

  int64_t slot = -1;
  for (size_t t = 0; t < upper.size(); ++t) {
    if (t == 0 || upper[t].i != upper[t - 1].i ||
        upper[t].j != upper[t - 1].j) {
      ++slot;
      Ar[slot] = upper[t].i;
      Axv[slot] = 0.0;
      Ap[upper[t].j + 1] += 1;  // per-column counts, prefix-summed below
    }
    entry->pos[upper[t].k] = slot;
  }
  for (int64_t j = 0; j < n; ++j) Ap[j + 1] += Ap[j];

  entry->A = A;
  entry->L = cholmod_analyze(A, &g_common);
  if (!entry->L || g_common.status < CHOLMOD_OK) {
    free_entry_locked(entry.get());
    *err = "cholmodjax: cholmod_analyze failed (status " +
           std::to_string(g_common.status) + ")";
    return nullptr;
  }

  uint64_t h = pattern_hash(Ai, Aj, nnz, n);
  auto& chain = g_registry[h];
  chain.push_back(std::move(entry));
  return chain.back().get();
}

static bool entry_matches(const PatternEntry* e, const int32_t* Ai,
                          const int32_t* Aj, int64_t nnz, int64_t n) {
  return e->n == n && e->Ai.size() == static_cast<size_t>(nnz) &&
         std::memcmp(e->Ai.data(), Ai, nnz * sizeof(int32_t)) == 0 &&
         std::memcmp(e->Aj.data(), Aj, nnz * sizeof(int32_t)) == 0;
}

static PatternEntry* get_or_create_entry_locked(const int32_t* Ai,
                                                const int32_t* Aj, int64_t nnz,
                                                int64_t n, std::string* err) {
  if (g_last_entry && entry_matches(g_last_entry, Ai, Aj, nnz, n))
    return g_last_entry;
  PatternEntry* found = nullptr;
  auto it = g_registry.find(pattern_hash(Ai, Aj, nnz, n));
  if (it != g_registry.end()) {
    for (auto& e : it->second) {
      if (entry_matches(e.get(), Ai, Aj, nnz, n)) {
        found = e.get();
        break;
      }
    }
  }
  if (!found) found = create_entry_locked(Ai, Aj, nnz, n, err);
  if (found) g_last_entry = found;
  return found;
}

// log|A| from the LL' factor diagonal (final_ll guarantees is_ll). Doubles
// as the positive-definiteness check: any nonpositive or NaN diagonal entry
// makes the result non-finite.
static double factor_logdet(const cholmod_factor* L) {
  double acc = 0.0;
  int64_t n = static_cast<int64_t>(L->n);
  if (L->is_super) {
    // Supernodal: diagonal of column j in supernode s sits at
    // x[px[s] + c*ld + c] with c = j - super[s], ld = pi[s+1] - pi[s].
    const int32_t* super = static_cast<const int32_t*>(L->super);
    const int32_t* pi = static_cast<const int32_t*>(L->pi);
    const int32_t* px = static_cast<const int32_t*>(L->px);
    const double* xv = static_cast<const double*>(L->x);
    int64_t nsuper = static_cast<int64_t>(L->nsuper);
    for (int64_t s = 0; s < nsuper; ++s) {
      int64_t ld = pi[s + 1] - pi[s];
      int64_t ncols = super[s + 1] - super[s];
      for (int64_t c = 0; c < ncols; ++c)
        acc += 2.0 * std::log(xv[px[s] + c * ld + c]);
    }
  } else {
    // Simplicial: the first stored entry of each column is the diagonal.
    const int32_t* p = static_cast<const int32_t*>(L->p);
    const double* xv = static_cast<const double*>(L->x);
    for (int64_t j = 0; j < n; ++j)
      acc += (L->is_ll ? 2.0 : 1.0) * std::log(xv[p[j]]);
  }
  return acc;
}

// Scatter COO values into A->x and (re)factorize, skipping the numeric
// factorization entirely when the values are identical to the previous call.
static bool factorize_locked(PatternEntry* e, const double* Ax, int64_t nnz,
                             std::string* err) {
  if (e->factored && e->last_Ax.size() == static_cast<size_t>(nnz) &&
      std::memcmp(e->last_Ax.data(), Ax, nnz * sizeof(double)) == 0)
    return true;

  double* Axv = static_cast<double*>(e->A->x);
  int64_t nnz_csc = static_cast<int64_t>(e->A->nzmax);
  std::memset(Axv, 0, nnz_csc * sizeof(double));
  for (int64_t k = 0; k < nnz; ++k)
    if (e->pos[k] >= 0) Axv[e->pos[k]] += Ax[k];

  g_common.status = CHOLMOD_OK;
  cholmod_factorize(e->A, e->L, &g_common);
  if (g_common.status == CHOLMOD_NOT_POSDEF) {
    e->factored = false;
    e->last_Ax.clear();
    *err = "cholmodjax: matrix is not positive definite (failure at column " +
           std::to_string(e->L->minor) + ")";
    return false;
  }
  if (g_common.status < CHOLMOD_OK) {
    e->factored = false;
    e->last_Ax.clear();
    *err = "cholmodjax: cholmod_factorize failed (status " +
           std::to_string(g_common.status) + ")";
    return false;
  }
  double ld = factor_logdet(e->L);
  if (!std::isfinite(ld)) {
    // Simplicial LDL' accepts indefinite matrices; the final_ll conversion
    // turns negative pivots into NaNs, which land here.
    e->factored = false;
    e->last_Ax.clear();
    *err = "cholmodjax: matrix is not positive definite";
    return false;
  }
  e->last_Ax.assign(Ax, Ax + nnz);
  e->logdet = ld;
  e->factored = true;
  e->factor_epoch++;  // invalidates any cached simplicial copy (Lldl)
  g_num_factorizations++;
  return true;
}

// Look up (or build) the pattern entry and (re)factorize it, throwing a Python
// exception on failure. For the numpy-callable core functions (used by non-JAX
// frontends such as the PyTensor Ops), which report errors via exceptions
// rather than ffi::Error. Caller must hold g_mutex.
static PatternEntry* prepare_entry_locked(const int32_t* Ai, const int32_t* Aj,
                                          const double* Ax, int64_t nnz,
                                          int64_t n) {
  std::string err;
  PatternEntry* e = get_or_create_entry_locked(Ai, Aj, nnz, n, &err);
  if (!e) throw std::runtime_error(err);
  if (!factorize_locked(e, Ax, nnz, &err)) throw std::runtime_error(err);
  return e;
}

// Solve L L' x = b for one right-hand side block using factor L and the
// entry's persistent solve2 workspaces. bdata/xdata are row-major (JAX
// layout); CHOLMOD is column-major, so multi-RHS blocks are transposed
// through `scratch` (reused across calls).
static ffi::Error solve_one_locked(PatternEntry* e, cholmod_factor* L, int mode,
                                   const double* bdata, int64_t n, int64_t nrhs,
                                   double* xdata, std::vector<double>* scratch) {
  const double* bcol = bdata;
  if (nrhs > 1) {
    scratch->resize(n * nrhs);
    for (int64_t i = 0; i < n; ++i)
      for (int64_t j = 0; j < nrhs; ++j)
        (*scratch)[i + j * n] = bdata[i * nrhs + j];
    bcol = scratch->data();
  }

  cholmod_dense B;
  std::memset(&B, 0, sizeof(B));
  B.nrow = n;
  B.ncol = nrhs;
  B.nzmax = n * nrhs;
  B.d = n;
  B.x = const_cast<double*>(bcol);
  B.xtype = CHOLMOD_REAL;
  B.dtype = CHOLMOD_DOUBLE;

  // cholmod_solve2 reuses the X/Y/E workspaces held in the entry, avoiding
  // cholmod_solve's per-call allocations.
  int ok = cholmod_solve2(mode, L, &B, nullptr, &e->Xwork, nullptr, &e->Ywork,
                          &e->Ework, &g_common);
  if (!ok || !e->Xwork || g_common.status < CHOLMOD_OK)
    return ffi::Error::Internal("cholmodjax: cholmod_solve failed (status " +
                                std::to_string(g_common.status) + ")");

  const double* Xx = static_cast<const double*>(e->Xwork->x);
  if (nrhs == 1) {
    std::memcpy(xdata, Xx, n * sizeof(double));
  } else {
    for (int64_t i = 0; i < n; ++i)
      for (int64_t j = 0; j < nrhs; ++j) xdata[i * nrhs + j] = Xx[i + j * n];
  }
  return ffi::Error::Success();
}

// ---------------------------------------------------------------------------
// solve handler:  (Ai, Aj, Ax, b; mode) -> x with x.shape == b.shape
// b may be (n,) or (n, nrhs). `mode` maps directly to CHOLMOD_A etc.
// ---------------------------------------------------------------------------

static ffi::Error SolveF64Impl(ffi::Buffer<ffi::S32> Ai,
                               ffi::Buffer<ffi::S32> Aj,
                               ffi::Buffer<ffi::F64> Ax,
                               ffi::Buffer<ffi::F64> b,
                               ffi::ResultBuffer<ffi::F64> x, int64_t mode) {
  auto bdims = b.dimensions();
  if (bdims.size() < 1 || bdims.size() > 2)
    return ffi::Error::InvalidArgument("cholmodjax: b must be 1D or 2D");
  int64_t n = bdims[0];
  int64_t nrhs = bdims.size() == 2 ? bdims[1] : 1;
  int64_t nnz = static_cast<int64_t>(Ai.element_count());
  if (static_cast<int64_t>(Aj.element_count()) != nnz ||
      static_cast<int64_t>(Ax.element_count()) != nnz)
    return ffi::Error::InvalidArgument(
        "cholmodjax: Ai, Aj, Ax must have the same length");
  if (mode < 0 || mode > 8)
    return ffi::Error::InvalidArgument("cholmodjax: invalid solve mode");

  std::lock_guard<std::mutex> lock(g_mutex);
  ensure_started_locked();

  std::string err;
  PatternEntry* e =
      get_or_create_entry_locked(Ai.typed_data(), Aj.typed_data(), nnz, n, &err);
  if (!e) return ffi::Error::InvalidArgument(err);
  if (!factorize_locked(e, Ax.typed_data(), nnz, &err))
    return ffi::Error::Internal(err);

  std::vector<double> scratch;
  return solve_one_locked(e, e->L, static_cast<int>(mode), b.typed_data(), n,
                          nrhs, x->typed_data(), &scratch);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CholmodSolveF64, SolveF64Impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Ai
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Aj
                                  .Arg<ffi::Buffer<ffi::F64>>()   // Ax
                                  .Arg<ffi::Buffer<ffi::F64>>()   // b
                                  .Ret<ffi::Buffer<ffi::F64>>()   // x
                                  .Attr<int64_t>("mode"));

// ---------------------------------------------------------------------------
// batched solve handler:  (Ai, Aj, Ax[B,nnz], b[B,n(,nrhs)]; mode) -> x[B,...]
// One FFI call solves a whole batch that shares a sparsity pattern, refactoring
// per element and reusing the cached analysis + solve workspaces. This is what
// jax.vmap(solve) lowers to (see the custom_vmap rule in Python), so the batch
// loop runs in C++ rather than as XLA per-iteration dispatch. Ai/Aj stay
// unbatched — the pattern is identical across the batch.
// ---------------------------------------------------------------------------

static ffi::Error SolveBatchedF64Impl(ffi::Buffer<ffi::S32> Ai,
                                      ffi::Buffer<ffi::S32> Aj,
                                      ffi::Buffer<ffi::F64> Ax,
                                      ffi::Buffer<ffi::F64> b,
                                      ffi::ResultBuffer<ffi::F64> x,
                                      int64_t mode) {
  auto bdims = b.dimensions();
  if (bdims.size() < 2 || bdims.size() > 3)
    return ffi::Error::InvalidArgument(
        "cholmodjax: batched b must be 2D or 3D");
  int64_t batch = bdims[0];
  int64_t n = bdims[1];
  int64_t nrhs = bdims.size() == 3 ? bdims[2] : 1;
  int64_t nnz = static_cast<int64_t>(Ai.element_count());
  auto axdims = Ax.dimensions();
  if (axdims.size() != 2 || axdims[0] != batch || axdims[1] != nnz)
    return ffi::Error::InvalidArgument(
        "cholmodjax: batched Ax must have shape (batch, nnz) matching b");
  if (static_cast<int64_t>(Aj.element_count()) != nnz)
    return ffi::Error::InvalidArgument(
        "cholmodjax: Ai, Aj must have the same length");
  if (mode < 0 || mode > 8)
    return ffi::Error::InvalidArgument("cholmodjax: invalid solve mode");

  std::lock_guard<std::mutex> lock(g_mutex);
  ensure_started_locked();

  std::string err;
  PatternEntry* e =
      get_or_create_entry_locked(Ai.typed_data(), Aj.typed_data(), nnz, n, &err);
  if (!e) return ffi::Error::InvalidArgument(err);

  const double* Axd = Ax.typed_data();
  const double* bd = b.typed_data();
  double* xd = x->typed_data();
  int64_t bstride = n * nrhs;
  std::vector<double> scratch;
  for (int64_t s = 0; s < batch; ++s) {
    if (!factorize_locked(e, Axd + s * nnz, nnz, &err))
      return ffi::Error::Internal(err);
    ffi::Error r =
        solve_one_locked(e, e->L, static_cast<int>(mode), bd + s * bstride, n,
                         nrhs, xd + s * bstride, &scratch);
    if (r.failure()) return r;
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CholmodSolveBatchedF64, SolveBatchedF64Impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Ai
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Aj
                                  .Arg<ffi::Buffer<ffi::F64>>()   // Ax [B,nnz]
                                  .Arg<ffi::Buffer<ffi::F64>>()   // b  [B,...]
                                  .Ret<ffi::Buffer<ffi::F64>>()   // x  [B,...]
                                  .Attr<int64_t>("mode"));

// ---------------------------------------------------------------------------
// logdet handler:  (Ai, Aj, Ax; n) -> scalar log|A|
// Shares the factorization cache with solve, so a solve followed by a logdet
// with identical values factorizes only once.
// ---------------------------------------------------------------------------

static ffi::Error LogdetF64Impl(ffi::Buffer<ffi::S32> Ai,
                                ffi::Buffer<ffi::S32> Aj,
                                ffi::Buffer<ffi::F64> Ax,
                                ffi::ResultBuffer<ffi::F64> out, int64_t n) {
  int64_t nnz = static_cast<int64_t>(Ai.element_count());
  if (static_cast<int64_t>(Aj.element_count()) != nnz ||
      static_cast<int64_t>(Ax.element_count()) != nnz)
    return ffi::Error::InvalidArgument(
        "cholmodjax: Ai, Aj, Ax must have the same length");

  std::lock_guard<std::mutex> lock(g_mutex);
  ensure_started_locked();

  std::string err;
  PatternEntry* e =
      get_or_create_entry_locked(Ai.typed_data(), Aj.typed_data(), nnz, n, &err);
  if (!e) return ffi::Error::InvalidArgument(err);
  if (!factorize_locked(e, Ax.typed_data(), nnz, &err))
    return ffi::Error::Internal(err);

  out->typed_data()[0] = e->logdet;
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CholmodLogdetF64, LogdetF64Impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Ai
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Aj
                                  .Arg<ffi::Buffer<ffi::F64>>()   // Ax
                                  .Ret<ffi::Buffer<ffi::F64>>()   // logdet
                                  .Attr<int64_t>("n"));

// ---------------------------------------------------------------------------
// factor-solve handler: factor A once, then apply K independent solve *chains*
// from that single factor, plus an optional logdet.
//
//   args:  Ai, Aj, Ax, b_0, ..., b_{K-1}   (each b_k is (n) or (n, nrhs_k))
//   rets:  x_0, ..., x_{K-1}[, logdet]      (x_k has the same shape as b_k)
//   attrs: mode_chain  — all K mode sequences flattened end to end
//          chain_lens  — length K; chain_lens[k] modes belong to b_k
//          want_logdet — 0/1; when 1 the final ret is the scalar log|A|
//          n           — matrix dimension (needed when K == 0)
//
// Each chain is applied left to right: x_k = m_last(... m_1(m_0(b_k))), where
// each m is a CHOLMOD solve system (MODE_*). This expresses e.g. the Gibbs
// Gaussian step from one factorization: mean = A^{-1} b via chain [MODE_A], and
// a draw's factor part P' L^{-T} z via chain [MODE_LT, MODE_PT].
// ---------------------------------------------------------------------------

// Apply all K solve chains for one already-factored system. bptr/xptr hold the
// K input/output row-major blocks; tmpA/tmpB are reused ping-pong scratch.
static ffi::Error apply_solves_locked(
    PatternEntry* e, cholmod_factor* L, int64_t n,
    const std::vector<const double*>& bptr, const std::vector<int64_t>& nrhs,
    const int64_t* mode_chain, const int64_t* chain_lens, int64_t K,
    const std::vector<double*>& xptr, std::vector<double>* scratch,
    std::vector<double>* tmpA, std::vector<double>* tmpB) {
  int64_t coff = 0;
  for (int64_t k = 0; k < K; ++k) {
    int64_t len = chain_lens[k];
    int64_t w = nrhs[k];
    if (len == 0) {
      // Empty chain: identity copy b_k -> x_k.
      std::memcpy(xptr[k], bptr[k], n * w * sizeof(double));
      continue;
    }
    const double* in = bptr[k];
    for (int64_t s = 0; s < len; ++s) {
      int64_t mode = mode_chain[coff + s];
      if (mode < 0 || mode > 8)
        return ffi::Error::InvalidArgument("cholmodjax: invalid solve mode");
      // Final step writes into the output buffer; intermediates ping-pong.
      double* out;
      if (s == len - 1)
        out = xptr[k];
      else {
        std::vector<double>* t = (s % 2 == 0) ? tmpA : tmpB;
        t->resize(n * w);
        out = t->data();
      }
      ffi::Error r = solve_one_locked(e, L, static_cast<int>(mode), in, n, w,
                                      out, scratch);
      if (r.failure()) return r;
      in = out;
    }
    coff += len;
  }
  return ffi::Error::Success();
}

static ffi::Error FactorSolveF64Impl(
    ffi::Buffer<ffi::S32> Ai, ffi::Buffer<ffi::S32> Aj, ffi::Buffer<ffi::F64> Ax,
    ffi::RemainingArgs rhs, ffi::RemainingRets rets,
    ffi::Span<const int64_t> mode_chain, ffi::Span<const int64_t> chain_lens,
    int64_t want_logdet, int64_t n) {
  int64_t nnz = static_cast<int64_t>(Ai.element_count());
  if (static_cast<int64_t>(Aj.element_count()) != nnz ||
      static_cast<int64_t>(Ax.element_count()) != nnz)
    return ffi::Error::InvalidArgument(
        "cholmodjax: Ai, Aj, Ax must have the same length");
  int64_t K = static_cast<int64_t>(chain_lens.size());
  if (static_cast<int64_t>(rhs.size()) != K)
    return ffi::Error::InvalidArgument(
        "cholmodjax: number of right-hand sides must match chain_lens");
  if (static_cast<int64_t>(rets.size()) != K + (want_logdet ? 1 : 0))
    return ffi::Error::InvalidArgument(
        "cholmodjax: number of results must match rhs count (+ logdet)");

  std::vector<const double*> bptr(K);
  std::vector<double*> xptr(K);
  std::vector<int64_t> nrhs(K);
  for (int64_t k = 0; k < K; ++k) {
    auto b = rhs.get<ffi::Buffer<ffi::F64>>(k);
    auto x = rets.get<ffi::Buffer<ffi::F64>>(k);
    if (b.has_error()) return b.error();
    if (x.has_error()) return x.error();
    auto bdims = b->dimensions();
    if (bdims.size() < 1 || bdims.size() > 2 || bdims[0] != n)
      return ffi::Error::InvalidArgument(
          "cholmodjax: each rhs must be (n) or (n, nrhs) with matching n");
    bptr[k] = b->typed_data();
    xptr[k] = (*x)->typed_data();
    nrhs[k] = bdims.size() == 2 ? bdims[1] : 1;
  }

  std::lock_guard<std::mutex> lock(g_mutex);
  ensure_started_locked();

  std::string err;
  PatternEntry* e =
      get_or_create_entry_locked(Ai.typed_data(), Aj.typed_data(), nnz, n, &err);
  if (!e) return ffi::Error::InvalidArgument(err);
  if (!factorize_locked(e, Ax.typed_data(), nnz, &err))
    return ffi::Error::Internal(err);

  std::vector<double> scratch, tmpA, tmpB;
  ffi::Error r = apply_solves_locked(e, e->L, n, bptr, nrhs, mode_chain.begin(),
                                     chain_lens.begin(), K, xptr, &scratch, &tmpA,
                                     &tmpB);
  if (r.failure()) return r;

  if (want_logdet) {
    auto ld = rets.get<ffi::Buffer<ffi::F64>>(K);
    if (ld.has_error()) return ld.error();
    (*ld)->typed_data()[0] = e->logdet;
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CholmodFactorSolveF64, FactorSolveF64Impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Ai
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Aj
                                  .Arg<ffi::Buffer<ffi::F64>>()   // Ax
                                  .RemainingArgs()                // b_0..b_{K-1}
                                  .RemainingRets()                // x_0..[,logdet]
                                  .Attr<ffi::Span<const int64_t>>("mode_chain")
                                  .Attr<ffi::Span<const int64_t>>("chain_lens")
                                  .Attr<int64_t>("want_logdet")
                                  .Attr<int64_t>("n"));

// Batched factor-solve: leading axis is the batch. Ax is (B, nnz); each rhs is
// (B, n[, nrhs_k]); results mirror that. The system is factored exactly ONCE
// per batch element, then every chain for that element is solved from it — so a
// vmapped multi-solve costs one factorization per element, not one per chain.
static ffi::Error FactorSolveBatchedF64Impl(
    ffi::Buffer<ffi::S32> Ai, ffi::Buffer<ffi::S32> Aj, ffi::Buffer<ffi::F64> Ax,
    ffi::RemainingArgs rhs, ffi::RemainingRets rets,
    ffi::Span<const int64_t> mode_chain, ffi::Span<const int64_t> chain_lens,
    int64_t want_logdet, int64_t n) {
  int64_t nnz = static_cast<int64_t>(Ai.element_count());
  auto axdims = Ax.dimensions();
  if (axdims.size() != 2 || axdims[1] != nnz)
    return ffi::Error::InvalidArgument(
        "cholmodjax: batched Ax must have shape (batch, nnz)");
  int64_t batch = axdims[0];
  int64_t K = static_cast<int64_t>(chain_lens.size());
  if (static_cast<int64_t>(rhs.size()) != K)
    return ffi::Error::InvalidArgument(
        "cholmodjax: number of right-hand sides must match chain_lens");
  if (static_cast<int64_t>(rets.size()) != K + (want_logdet ? 1 : 0))
    return ffi::Error::InvalidArgument(
        "cholmodjax: number of results must match rhs count (+ logdet)");

  // Per-rhs element strides and per-batch base pointers.
  std::vector<const double*> bbase(K);
  std::vector<double*> xbase(K);
  std::vector<int64_t> nrhs(K), bstride(K);
  for (int64_t k = 0; k < K; ++k) {
    auto b = rhs.get<ffi::Buffer<ffi::F64>>(k);
    auto x = rets.get<ffi::Buffer<ffi::F64>>(k);
    if (b.has_error()) return b.error();
    if (x.has_error()) return x.error();
    auto bdims = b->dimensions();
    if (bdims.size() < 2 || bdims.size() > 3 || bdims[0] != batch ||
        bdims[1] != n)
      return ffi::Error::InvalidArgument(
          "cholmodjax: each batched rhs must be (batch, n[, nrhs])");
    nrhs[k] = bdims.size() == 3 ? bdims[2] : 1;
    bstride[k] = n * nrhs[k];
    bbase[k] = b->typed_data();
    xbase[k] = (*x)->typed_data();
  }
  double* ldbase = nullptr;
  if (want_logdet) {
    auto ld = rets.get<ffi::Buffer<ffi::F64>>(K);
    if (ld.has_error()) return ld.error();
    ldbase = (*ld)->typed_data();
  }

  std::lock_guard<std::mutex> lock(g_mutex);
  ensure_started_locked();

  std::string err;
  PatternEntry* e =
      get_or_create_entry_locked(Ai.typed_data(), Aj.typed_data(), nnz, n, &err);
  if (!e) return ffi::Error::InvalidArgument(err);

  const double* Axd = Ax.typed_data();
  std::vector<double> scratch, tmpA, tmpB;
  std::vector<const double*> bptr(K);
  std::vector<double*> xptr(K);
  for (int64_t s = 0; s < batch; ++s) {
    if (!factorize_locked(e, Axd + s * nnz, nnz, &err))
      return ffi::Error::Internal(err);
    for (int64_t k = 0; k < K; ++k) {
      bptr[k] = bbase[k] + s * bstride[k];
      xptr[k] = xbase[k] + s * bstride[k];
    }
    ffi::Error r =
        apply_solves_locked(e, e->L, n, bptr, nrhs, mode_chain.begin(),
                            chain_lens.begin(), K, xptr, &scratch, &tmpA, &tmpB);
    if (r.failure()) return r;
    if (want_logdet) ldbase[s] = e->logdet;
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CholmodFactorSolveBatchedF64,
                              FactorSolveBatchedF64Impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Ai
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Aj
                                  .Arg<ffi::Buffer<ffi::F64>>()   // Ax [B,nnz]
                                  .RemainingArgs()                // b_0..b_{K-1}
                                  .RemainingRets()                // x_0..[,logdet]
                                  .Attr<ffi::Span<const int64_t>>("mode_chain")
                                  .Attr<ffi::Span<const int64_t>>("chain_lens")
                                  .Attr<int64_t>("want_logdet")
                                  .Attr<int64_t>("n"));

// ---------------------------------------------------------------------------
// Simplicial LDL' base factor (Lldl). cholmod_updown and the selected-inverse
// recurrence both need an LDL' (not LL', not supernodal) factor. Lldl is a
// persistent simplicial LDL' copy of the base factor L, rebuilt only when the
// base is refactored (tracked by epoch), so the LL'->LDL' conversion is paid
// once per base change rather than once per call.
// ---------------------------------------------------------------------------

static ffi::Error ensure_ldl_locked(PatternEntry* e) {
  if (e->Lldl && e->ldl_epoch == e->factor_epoch) return ffi::Error::Success();
  if (e->Lldl) cholmod_free_factor(&e->Lldl, &g_common);
  e->Lldl = cholmod_copy_factor(e->L, &g_common);
  if (!e->Lldl)
    return ffi::Error::Internal("cholmodjax: cholmod_copy_factor failed");
  if (!cholmod_change_factor(CHOLMOD_REAL, /*to_ll=*/0, /*to_super=*/0,
                             /*to_packed=*/1, /*to_monotonic=*/1, e->Lldl,
                             &g_common))
    return ffi::Error::Internal("cholmodjax: cholmod_change_factor failed");
  e->ldl_epoch = e->factor_epoch;
  return ffi::Error::Success();
}

// ---------------------------------------------------------------------------
// selected-inverse handler:  (Ai, Aj, Ax; n) -> z[nnz]
//     z[k] = (A^{-1})[Ai[k], Aj[k]]
//
// Computes the entries of A^{-1} that lie in the pattern of A (a subset of the
// factor's fill pattern) via Takahashi's recurrence on the LDL' factor, without
// forming the dense inverse. This is exactly what logdet's reverse-mode rule
// needs: d log|A| / d A = A^{-1}, so d log|A| / d Ax[k] is (A^{-1}) at the COO
// position (Ai[k], Aj[k]) (see the custom_vjp in Python for the symmetric
// doubling of off-diagonal entries).
//
// Recurrence (A = P' L D L' P, L unit lower-triangular, D diagonal; work in the
// permuted space, then map back through Perm). Processing columns j = n-1..0,
// with below-diagonal pattern rows r_0<...<r_{p-1} of column j and l_b=L[r_b,j]:
//     Z[r_a, j] = - sum_b l_b * Z[r_a, r_b]           (selected inverse Z)
//     Z[j, j]   = 1/D[j] - sum_a l_a * Z[r_a, j]
// The pairs (r_a, r_b) are all in the factor's pattern (fill closure), so the
// recurrence stays within the stored structure.
// ---------------------------------------------------------------------------

static ffi::Error selinv_locked(PatternEntry* e, int64_t nnz, double* z) {
  ffi::Error r = ensure_ldl_locked(e);
  if (r.failure()) return r;
  cholmod_factor* L = e->Lldl;
  int64_t n = static_cast<int64_t>(L->n);
  const int32_t* Lp = static_cast<const int32_t*>(L->p);
  const int32_t* Li = static_cast<const int32_t*>(L->i);
  const double* Lx = static_cast<const double*>(L->x);
  const int32_t* Perm = static_cast<const int32_t*>(L->Perm);

  std::vector<double> Z(Lp[n], 0.0);  // selected inverse, same structure as L
  std::vector<double> work(n, 0.0);   // dense scatter workspace (kept zeroed)
  std::vector<double> acc;

  for (int64_t j = n - 1; j >= 0; --j) {
    int64_t ps = Lp[j], pe = Lp[j + 1];
    int64_t cnt = pe - ps - 1;  // below-diagonal entries of column j
    // acc[a] = sum_b L[r_b, j] * Z[r_a, r_b], accumulated symmetrically so that
    // each column of Z is scattered into `work` exactly once.
    acc.assign(cnt, 0.0);
    for (int64_t b = 0; b < cnt; ++b) {
      int64_t c = Li[ps + 1 + b];  // r_b: column of Z to scatter
      double lb = Lx[ps + 1 + b];
      int64_t cs = Lp[c], ce = Lp[c + 1];
      work[c] = Z[cs];  // Z[c, c]
      for (int64_t t = cs + 1; t < ce; ++t) work[Li[t]] = Z[t];
      for (int64_t a = b; a < cnt; ++a) {
        double za = work[Li[ps + 1 + a]];  // Z[r_a, r_b], r_a >= r_b (stored)
        acc[a] += lb * za;
        if (a != b) acc[b] += Lx[ps + 1 + a] * za;  // symmetric partner
      }
      work[c] = 0.0;  // restore workspace to all-zero
      for (int64_t t = cs + 1; t < ce; ++t) work[Li[t]] = 0.0;
    }
    double diag = 1.0 / Lx[ps];  // 1/D[j]
    for (int64_t a = 0; a < cnt; ++a) {
      double zaj = -acc[a];
      Z[ps + 1 + a] = zaj;
      diag -= Lx[ps + 1 + a] * zaj;
    }
    Z[ps] = diag;
  }

  // Z is A^{-1} in the permuted space: Z[u,v] = (A^{-1})[Perm[u], Perm[v]].
  // Map each COO position back through the inverse permutation.
  std::vector<int32_t> iperm(n);
  for (int64_t k = 0; k < n; ++k) iperm[Perm[k]] = static_cast<int32_t>(k);
  const int32_t* Ai = e->Ai.data();
  const int32_t* Aj = e->Aj.data();
  for (int64_t k = 0; k < nnz; ++k) {
    int64_t u = iperm[Ai[k]], v = iperm[Aj[k]];
    if (u < v) std::swap(u, v);  // stored entry lives in column min, row max
    double val = 0.0;
    if (u == v) {
      val = Z[Lp[v]];
    } else {
      for (int64_t t = Lp[v] + 1, te = Lp[v + 1]; t < te; ++t)
        if (Li[t] == u) {
          val = Z[t];
          break;
        }
    }
    z[k] = val;
  }
  return ffi::Error::Success();
}

static ffi::Error SelinvF64Impl(ffi::Buffer<ffi::S32> Ai,
                                ffi::Buffer<ffi::S32> Aj,
                                ffi::Buffer<ffi::F64> Ax,
                                ffi::ResultBuffer<ffi::F64> z, int64_t n) {
  int64_t nnz = static_cast<int64_t>(Ai.element_count());
  if (static_cast<int64_t>(Aj.element_count()) != nnz ||
      static_cast<int64_t>(Ax.element_count()) != nnz)
    return ffi::Error::InvalidArgument(
        "cholmodjax: Ai, Aj, Ax must have the same length");

  std::lock_guard<std::mutex> lock(g_mutex);
  ensure_started_locked();

  std::string err;
  PatternEntry* e =
      get_or_create_entry_locked(Ai.typed_data(), Aj.typed_data(), nnz, n, &err);
  if (!e) return ffi::Error::InvalidArgument(err);
  if (!factorize_locked(e, Ax.typed_data(), nnz, &err))
    return ffi::Error::Internal(err);

  return selinv_locked(e, nnz, z->typed_data());
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CholmodSelinvF64, SelinvF64Impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Ai
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Aj
                                  .Arg<ffi::Buffer<ffi::F64>>()   // Ax
                                  .Ret<ffi::Buffer<ffi::F64>>()   // z [nnz]
                                  .Attr<int64_t>("n"));

// ---------------------------------------------------------------------------
// updown-solve handler:  (Ai, Aj, Ax, C[n,k], b[n(,nrhs)]; mode, downdate)
//     -> x solving (A ± C C') x = b, and out_logdet = log|A ± C C'|.
//
// The base matrix A is factored once (cached). Each call rebuilds a simplicial
// LDL' copy of that factor, applies cholmod_updown for the rank-k modification
// C C', and solves — much cheaper than refactoring A ± C C' from scratch when A
// is held fixed and only the low-rank term C varies. `update=1` adds C C',
// `downdate=1` (downdate flag) subtracts it. The base cached factor is never
// mutated, so the op is a pure function of its inputs.
// C is dense (n, k), row-major (JAX layout).
// ---------------------------------------------------------------------------

static ffi::Error UpdownSolveF64Impl(ffi::Buffer<ffi::S32> Ai,
                                     ffi::Buffer<ffi::S32> Aj,
                                     ffi::Buffer<ffi::F64> Ax,
                                     ffi::Buffer<ffi::F64> C,
                                     ffi::Buffer<ffi::F64> b,
                                     ffi::ResultBuffer<ffi::F64> x,
                                     ffi::ResultBuffer<ffi::F64> out_logdet,
                                     int64_t mode, int64_t downdate) {
  auto bdims = b.dimensions();
  if (bdims.size() < 1 || bdims.size() > 2)
    return ffi::Error::InvalidArgument("cholmodjax: b must be 1D or 2D");
  int64_t n = bdims[0];
  int64_t nrhs = bdims.size() == 2 ? bdims[1] : 1;
  int64_t nnz = static_cast<int64_t>(Ai.element_count());
  if (static_cast<int64_t>(Aj.element_count()) != nnz ||
      static_cast<int64_t>(Ax.element_count()) != nnz)
    return ffi::Error::InvalidArgument(
        "cholmodjax: Ai, Aj, Ax must have the same length");
  auto cdims = C.dimensions();
  if (cdims.size() != 2 || cdims[0] != n)
    return ffi::Error::InvalidArgument(
        "cholmodjax: C must have shape (n, k) matching b's dimension n");
  int64_t k = cdims[1];
  if (mode < 0 || mode > 8)
    return ffi::Error::InvalidArgument("cholmodjax: invalid solve mode");

  std::lock_guard<std::mutex> lock(g_mutex);
  ensure_started_locked();

  std::string err;
  PatternEntry* e =
      get_or_create_entry_locked(Ai.typed_data(), Aj.typed_data(), nnz, n, &err);
  if (!e) return ffi::Error::InvalidArgument(err);
  if (!factorize_locked(e, Ax.typed_data(), nnz, &err))
    return ffi::Error::Internal(err);

  // cholmod_updown requires a simplicial LDL' factor (it cannot update a
  // supernodal or LL' factor). Lldl is a persistent simplicial LDL' copy of the
  // base factor (rebuilt only on refactor); the per-call working copy Lupd
  // (which updown mutates) is a plain simplicial->simplicial copy of Lldl,
  // leaving the base factor pristine.
  ffi::Error ldl_err = ensure_ldl_locked(e);
  if (ldl_err.failure()) return ldl_err;

  if (e->Lupd) cholmod_free_factor(&e->Lupd, &g_common);
  e->Lupd = cholmod_copy_factor(e->Lldl, &g_common);
  if (!e->Lupd)
    return ffi::Error::Internal("cholmodjax: cholmod_copy_factor failed");

  // Build sparse C (n x k) from the dense, row-major input via a column-major
  // temporary. cholmod_updown works in the factor's permuted space
  // (L D L' = P A P'), so C's rows must be permuted by L->Perm — permuted row
  // kk takes original row Perm[kk]. cholmod_solve then applies P/P' itself, so
  // the returned solution is in the original ordering. dense_to_sparse drops
  // explicit zeros, so a sparse update column yields a sparse C and cheap updown.
  cholmod_dense Cd;
  std::memset(&Cd, 0, sizeof(Cd));
  Cd.nrow = n;
  Cd.ncol = k;
  Cd.nzmax = n * k;
  Cd.d = n;
  Cd.xtype = CHOLMOD_REAL;
  Cd.dtype = CHOLMOD_DOUBLE;
  std::vector<double> Ccol(n * k);
  const double* Cin = C.typed_data();
  const int32_t* Perm = static_cast<const int32_t*>(e->Lupd->Perm);
  for (int64_t kk = 0; kk < n; ++kk) {
    int64_t orig = Perm[kk];
    for (int64_t j = 0; j < k; ++j) Ccol[kk + j * n] = Cin[orig * k + j];
  }
  Cd.x = Ccol.data();

  cholmod_sparse* Cs = cholmod_dense_to_sparse(&Cd, /*values=*/1, &g_common);
  if (!Cs)
    return ffi::Error::Internal("cholmodjax: cholmod_dense_to_sparse failed");

  int ok = cholmod_updown(downdate ? 0 : 1, Cs, e->Lupd, &g_common);
  cholmod_free_sparse(&Cs, &g_common);
  if (!ok || g_common.status < CHOLMOD_OK)
    return ffi::Error::Internal("cholmodjax: cholmod_updown failed (status " +
                                std::to_string(g_common.status) + ")");

  double ld = factor_logdet(e->Lupd);
  if (!std::isfinite(ld))
    return ffi::Error::Internal(
        "cholmodjax: updated matrix A ± C C' is not positive definite");
  out_logdet->typed_data()[0] = ld;

  std::vector<double> scratch;
  return solve_one_locked(e, e->Lupd, static_cast<int>(mode), b.typed_data(), n,
                          nrhs, x->typed_data(), &scratch);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CholmodUpdownSolveF64, UpdownSolveF64Impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Ai
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Aj
                                  .Arg<ffi::Buffer<ffi::F64>>()   // Ax
                                  .Arg<ffi::Buffer<ffi::F64>>()   // C [n,k]
                                  .Arg<ffi::Buffer<ffi::F64>>()   // b
                                  .Ret<ffi::Buffer<ffi::F64>>()   // x
                                  .Ret<ffi::Buffer<ffi::F64>>()   // logdet
                                  .Attr<int64_t>("mode")
                                  .Attr<int64_t>("downdate"));

// ---------------------------------------------------------------------------
// Numpy-callable core (framework-agnostic, no XLA). These call the same cached
// CHOLMOD core as the FFI handlers but take and return plain numpy arrays, so
// non-JAX frontends — the PyTensor Ops in cholmodjax.pytensor — can reach the
// solver and its selected-inverse gradient without going through XLA. Errors
// are raised as Python exceptions.
// ---------------------------------------------------------------------------

using ArrI32 =
    nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using ArrF64 = nb::ndarray<const double, nb::c_contig, nb::device::cpu>;

static void check_coo_np(const ArrI32& Ai, const ArrI32& Aj, const ArrF64& Ax,
                         int64_t nnz) {
  if (static_cast<int64_t>(Aj.shape(0)) != nnz || Ax.ndim() != 1 ||
      static_cast<int64_t>(Ax.shape(0)) != nnz)
    throw std::runtime_error(
        "cholmodjax: Ai, Aj, Ax must be 1D with the same length");
}

static nb::ndarray<nb::numpy, double> solve_np(ArrI32 Ai, ArrI32 Aj, ArrF64 Ax,
                                               ArrF64 b, int64_t mode) {
  int64_t nnz = static_cast<int64_t>(Ai.shape(0));
  check_coo_np(Ai, Aj, Ax, nnz);
  if (b.ndim() < 1 || b.ndim() > 2)
    throw std::runtime_error("cholmodjax: b must be 1D or 2D");
  if (mode < 0 || mode > 8)
    throw std::runtime_error("cholmodjax: invalid solve mode");
  int64_t n = static_cast<int64_t>(b.shape(0));
  int64_t nrhs = b.ndim() == 2 ? static_cast<int64_t>(b.shape(1)) : 1;

  double* out = new double[static_cast<size_t>(n) * nrhs];
  nb::capsule owner(out,
                    [](void* p) noexcept { delete[] static_cast<double*>(p); });
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    ensure_started_locked();
    PatternEntry* e =
        prepare_entry_locked(Ai.data(), Aj.data(), Ax.data(), nnz, n);
    std::vector<double> scratch;
    ffi::Error r = solve_one_locked(e, e->L, static_cast<int>(mode), b.data(), n,
                                    nrhs, out, &scratch);
    if (r.failure()) throw std::runtime_error("cholmodjax: solve failed");
  }
  if (b.ndim() == 2)
    return nb::ndarray<nb::numpy, double>(
        out, {static_cast<size_t>(n), static_cast<size_t>(nrhs)}, owner);
  return nb::ndarray<nb::numpy, double>(out, {static_cast<size_t>(n)}, owner);
}

static double logdet_np(ArrI32 Ai, ArrI32 Aj, ArrF64 Ax, int64_t n) {
  int64_t nnz = static_cast<int64_t>(Ai.shape(0));
  check_coo_np(Ai, Aj, Ax, nnz);
  std::lock_guard<std::mutex> lock(g_mutex);
  ensure_started_locked();
  PatternEntry* e =
      prepare_entry_locked(Ai.data(), Aj.data(), Ax.data(), nnz, n);
  return e->logdet;
}

static nb::ndarray<nb::numpy, double> selinv_np(ArrI32 Ai, ArrI32 Aj, ArrF64 Ax,
                                                int64_t n) {
  int64_t nnz = static_cast<int64_t>(Ai.shape(0));
  check_coo_np(Ai, Aj, Ax, nnz);
  double* out = new double[nnz > 0 ? static_cast<size_t>(nnz) : 1];
  nb::capsule owner(out,
                    [](void* p) noexcept { delete[] static_cast<double*>(p); });
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    ensure_started_locked();
    PatternEntry* e =
        prepare_entry_locked(Ai.data(), Aj.data(), Ax.data(), nnz, n);
    ffi::Error r = selinv_locked(e, nnz, out);
    if (r.failure()) throw std::runtime_error("cholmodjax: selinv failed");
  }
  return nb::ndarray<nb::numpy, double>(out, {static_cast<size_t>(nnz)}, owner);
}

// ---------------------------------------------------------------------------
// nanobind module
// ---------------------------------------------------------------------------

NB_MODULE(cholmodjax_cpp, m) {
  m.doc() = "CHOLMOD sparse Cholesky as XLA FFI custom calls";

  m.def("solve_f64_capsule", []() {
    return nb::capsule(reinterpret_cast<void*>(CholmodSolveF64));
  });
  m.def("solve_batched_f64_capsule", []() {
    return nb::capsule(reinterpret_cast<void*>(CholmodSolveBatchedF64));
  });
  m.def("logdet_f64_capsule", []() {
    return nb::capsule(reinterpret_cast<void*>(CholmodLogdetF64));
  });
  m.def("selinv_f64_capsule", []() {
    return nb::capsule(reinterpret_cast<void*>(CholmodSelinvF64));
  });
  m.def("updown_solve_f64_capsule", []() {
    return nb::capsule(reinterpret_cast<void*>(CholmodUpdownSolveF64));
  });
  m.def("factor_solve_f64_capsule", []() {
    return nb::capsule(reinterpret_cast<void*>(CholmodFactorSolveF64));
  });
  m.def("factor_solve_batched_f64_capsule", []() {
    return nb::capsule(reinterpret_cast<void*>(CholmodFactorSolveBatchedF64));
  });

  // Numpy-callable core, for non-JAX frontends (cholmodjax.pytensor).
  m.def("solve_np", &solve_np);
  m.def("logdet_np", &logdet_np);
  m.def("selinv_np", &selinv_np);

  // Total numeric (re)factorizations performed, for tests/introspection.
  m.def("factorization_count", []() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_num_factorizations;
  });

  m.def("clear_cache", []() {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_last_entry = nullptr;
    if (!g_started) return;
    for (auto& [h, chain] : g_registry)
      for (auto& e : chain) free_entry_locked(e.get());
    g_registry.clear();
  });

  m.def("cache_size", []() {
    std::lock_guard<std::mutex> lock(g_mutex);
    size_t count = 0;
    for (auto& [h, chain] : g_registry) count += chain.size();
    return count;
  });

  // 0 = CHOLMOD_SIMPLICIAL, 1 = CHOLMOD_AUTO, 2 = CHOLMOD_SUPERNODAL.
  // Only affects patterns analyzed after the call; callers clear the cache.
  m.def("set_supernodal", [](int v) {
    std::lock_guard<std::mutex> lock(g_mutex);
    ensure_started_locked();
    g_common.supernodal = v;
  });
}
