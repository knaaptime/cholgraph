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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
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
  return true;
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
  // supernodal or LL' factor). Maintain Lldl, a persistent simplicial LDL' copy
  // of the base factor, rebuilt only when the base is refactored — so the
  // LL'->LDL' conversion is paid once per base change, not once per call. The
  // per-call working copy Lupd (which updown mutates) is then a plain
  // simplicial->simplicial copy of Lldl, leaving the base factor pristine.
  if (!e->Lldl || e->ldl_epoch != e->factor_epoch) {
    if (e->Lldl) cholmod_free_factor(&e->Lldl, &g_common);
    e->Lldl = cholmod_copy_factor(e->L, &g_common);
    if (!e->Lldl)
      return ffi::Error::Internal("cholmodjax: cholmod_copy_factor failed");
    if (!cholmod_change_factor(CHOLMOD_REAL, /*to_ll=*/0, /*to_super=*/0,
                               /*to_packed=*/1, /*to_monotonic=*/1, e->Lldl,
                               &g_common))
      return ffi::Error::Internal("cholmodjax: cholmod_change_factor failed");
    e->ldl_epoch = e->factor_epoch;
  }

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
  m.def("updown_solve_f64_capsule", []() {
    return nb::capsule(reinterpret_cast<void*>(CholmodUpdownSolveF64));
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
