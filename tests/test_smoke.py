"""Tests for cholgraph: correctness, jit, scan, vmap, AD, logdet, errors."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp

import cholgraph


def grid_laplacian(k, seed=0):
    """SPD test matrix: 2D grid Laplacian + random diagonal, n = k*k."""
    rng = np.random.default_rng(seed)
    L = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(k, k))
    A = sp.kronsum(L, L) + sp.diags(rng.uniform(0.5, 1.5, k * k))
    A = sp.coo_matrix(A)
    return (
        A.row.astype(np.int32),
        A.col.astype(np.int32),
        A.data.astype(np.float64),
        A.toarray(),
    )


@pytest.fixture
def spd():
    return grid_laplacian(8)


class TestSolveCorrectness:
    def test_full_symmetric_coo(self, spd):
        Ai, Aj, Ax, A = spd
        b = np.arange(1.0, A.shape[0] + 1.0)
        x = cholgraph.solve(Ai, Aj, Ax, b)
        np.testing.assert_allclose(x, np.linalg.solve(A, b), rtol=1e-10)

    def test_upper_triangle_only(self, spd):
        Ai, Aj, Ax, A = spd
        keep = Ai <= Aj
        b = np.arange(1.0, A.shape[0] + 1.0)
        x = cholgraph.solve(Ai[keep], Aj[keep], Ax[keep], b)
        np.testing.assert_allclose(x, np.linalg.solve(A, b), rtol=1e-10)

    def test_multi_rhs(self, spd):
        Ai, Aj, Ax, A = spd
        rng = np.random.default_rng(1)
        B = rng.normal(size=(A.shape[0], 5))
        X = cholgraph.solve(Ai, Aj, Ax, B)
        assert X.shape == B.shape
        np.testing.assert_allclose(X, np.linalg.solve(A, B), rtol=1e-10)

    def test_duplicates_summed(self):
        # A = [[2, 1], [1, 3]] with the diagonal split across duplicates.
        Ai = np.array([0, 0, 0, 1, 1], dtype=np.int32)
        Aj = np.array([0, 0, 1, 1, 1], dtype=np.int32)
        Ax = np.array([1.0, 1.0, 1.0, 2.0, 1.0])
        b = np.array([1.0, 2.0])
        x = cholgraph.solve(Ai, Aj, Ax, b)
        np.testing.assert_allclose(
            x, np.linalg.solve(np.array([[2.0, 1.0], [1.0, 3.0]]), b), rtol=1e-12
        )


class TestJIT:
    def test_jit_compiles_and_is_correct(self, spd):
        Ai, Aj, Ax, A = spd
        b = np.arange(1.0, A.shape[0] + 1.0)

        @jax.jit
        def f(Ax, b):
            return cholgraph.solve(Ai, Aj, Ax, b)

        np.testing.assert_allclose(f(Ax, b), np.linalg.solve(A, b), rtol=1e-10)

    def test_jit_changing_values_same_pattern(self, spd):
        """The Gibbs-sampler pattern: same sparsity, new values every call."""
        Ai, Aj, Ax, A = spd
        n = A.shape[0]
        b = np.ones(n)
        f = jax.jit(lambda Ax, b: cholgraph.solve(Ai, Aj, Ax, b))
        for scale in (1.0, 2.5, 0.7):
            np.testing.assert_allclose(
                f(scale * Ax, b), np.linalg.solve(scale * A, b), rtol=1e-10
            )

    def test_lax_scan_loop(self, spd):
        Ai, Aj, Ax, A = spd
        n = A.shape[0]
        b = np.ones(n)

        @jax.jit
        def run(Ax0):
            def step(carry, scale):
                x = cholgraph.solve(Ai, Aj, Ax0 * scale, b)
                return carry + x.sum(), x.sum()

            return jax.lax.scan(step, 0.0, jnp.array([1.0, 2.0, 4.0]))

        total, sums = run(jnp.asarray(Ax))
        expected = [np.linalg.solve(s * A, b).sum() for s in (1.0, 2.0, 4.0)]
        np.testing.assert_allclose(sums, expected, rtol=1e-10)
        np.testing.assert_allclose(total, sum(expected), rtol=1e-10)


class TestBatching:
    def test_vmap_over_b(self, spd):
        Ai, Aj, Ax, A = spd
        rng = np.random.default_rng(2)
        bs = rng.normal(size=(4, A.shape[0]))
        xs = jax.vmap(lambda b: cholgraph.solve(Ai, Aj, Ax, b))(bs)
        np.testing.assert_allclose(xs, np.linalg.solve(A, bs.T).T, rtol=1e-10)

    def test_vmap_over_values(self, spd):
        Ai, Aj, Ax, A = spd
        b = np.ones(A.shape[0])
        scales = np.array([1.0, 3.0])
        Axs = scales[:, None] * Ax
        xs = jax.vmap(lambda Ax: cholgraph.solve(Ai, Aj, Ax, b))(Axs)
        for s, x in zip(scales, xs):
            np.testing.assert_allclose(x, np.linalg.solve(s * A, b), rtol=1e-10)

    def test_vmap_over_both(self, spd):
        Ai, Aj, Ax, A = spd
        rng = np.random.default_rng(4)
        scales = np.array([1.0, 2.0, 0.5])
        Axs = scales[:, None] * Ax
        bs = rng.normal(size=(3, A.shape[0]))
        xs = jax.jit(jax.vmap(lambda Ax, b: cholgraph.solve(Ai, Aj, Ax, b)))(Axs, bs)
        for i, s in enumerate(scales):
            np.testing.assert_allclose(xs[i], np.linalg.solve(s * A, bs[i]), rtol=1e-10)

    def test_vmap_multi_rhs(self, spd):
        Ai, Aj, Ax, A = spd
        rng = np.random.default_rng(5)
        Bs = rng.normal(size=(3, A.shape[0], 2))
        xs = jax.vmap(lambda B: cholgraph.solve(Ai, Aj, Ax, B))(Bs)
        for i in range(3):
            np.testing.assert_allclose(xs[i], np.linalg.solve(A, Bs[i]), rtol=1e-10)

    def test_vmap_lowers_to_single_batched_call(self, spd):
        """vmap must route to the native batched FFI call, exactly once."""
        Ai, Aj, Ax, A = spd
        bs = np.ones((4, A.shape[0]))
        f = jax.jit(jax.vmap(lambda b: cholgraph.solve(Ai, Aj, Ax, b)))
        hlo = f.lower(bs).compile().as_text()
        assert hlo.count("cholgraph_solve_batched_f64") == 1

    def test_vmap_of_grad(self, spd):
        """vmap composed with grad still batches and stays correct."""
        Ai, Aj, Ax, A = spd
        n = A.shape[0]
        b = np.ones(n)
        scales = np.array([1.0, 2.0, 3.0])
        Axs = scales[:, None] * Ax

        def loss(Ax):
            return (cholgraph.solve(Ai, Aj, Ax, b) ** 2).sum()

        gs = jax.vmap(jax.grad(loss))(Axs)
        for i in range(len(scales)):
            g1 = jax.grad(loss)(Axs[i])
            np.testing.assert_allclose(gs[i], g1, rtol=1e-10)


class TestUpdown:
    def _dense_from_coo(self, Ai, Aj, Ax, A):
        return A  # spd fixture already provides the dense form

    def test_rank1_update(self, spd):
        Ai, Aj, Ax, A = spd
        rng = np.random.default_rng(6)
        n = A.shape[0]
        c = rng.normal(size=n)
        b = rng.normal(size=n)
        x = cholgraph.update_solve(Ai, Aj, Ax, c, b)
        np.testing.assert_allclose(x, np.linalg.solve(A + np.outer(c, c), b), rtol=1e-9)

    def test_rank_k_update(self, spd):
        Ai, Aj, Ax, A = spd
        rng = np.random.default_rng(7)
        n = A.shape[0]
        C = rng.normal(size=(n, 4)) * 0.3
        b = rng.normal(size=n)
        x = cholgraph.update_solve(Ai, Aj, Ax, C, b)
        np.testing.assert_allclose(x, np.linalg.solve(A + C @ C.T, b), rtol=1e-9)

    def test_downdate(self, spd):
        Ai, Aj, Ax, A = spd
        rng = np.random.default_rng(8)
        n = A.shape[0]
        c = rng.normal(size=n) * 0.1  # small so A - cc' stays SPD
        b = rng.normal(size=n)
        x = cholgraph.update_solve(Ai, Aj, Ax, c, b, downdate=True)
        np.testing.assert_allclose(x, np.linalg.solve(A - np.outer(c, c), b), rtol=1e-9)

    def test_logdet(self, spd):
        Ai, Aj, Ax, A = spd
        rng = np.random.default_rng(9)
        n = A.shape[0]
        c = rng.normal(size=n)
        b = rng.normal(size=n)
        x, ld = cholgraph.update_solve(Ai, Aj, Ax, c, b, return_logdet=True)
        np.testing.assert_allclose(
            float(ld), np.linalg.slogdet(A + np.outer(c, c))[1], rtol=1e-10
        )

    def test_multi_rhs(self, spd):
        Ai, Aj, Ax, A = spd
        rng = np.random.default_rng(10)
        n = A.shape[0]
        c = rng.normal(size=n)
        B = rng.normal(size=(n, 3))
        X = cholgraph.update_solve(Ai, Aj, Ax, c, B)
        np.testing.assert_allclose(X, np.linalg.solve(A + np.outer(c, c), B), rtol=1e-9)

    def test_under_jit(self, spd):
        Ai, Aj, Ax, A = spd
        rng = np.random.default_rng(11)
        n = A.shape[0]
        c = rng.normal(size=n)
        b = rng.normal(size=n)
        f = jax.jit(lambda Ax, c, b: cholgraph.update_solve(Ai, Aj, Ax, c, b))
        np.testing.assert_allclose(
            f(Ax, c, b), np.linalg.solve(A + np.outer(c, c), b), rtol=1e-9
        )

    def test_base_matches_zero_update(self, spd):
        """A zero-column update must reproduce the plain solve."""
        Ai, Aj, Ax, A = spd
        n = A.shape[0]
        b = np.arange(1.0, n + 1.0)
        x = cholgraph.update_solve(Ai, Aj, Ax, np.zeros(n), b)
        np.testing.assert_allclose(x, np.linalg.solve(A, b), rtol=1e-10)

    def test_changing_base_rebuilds_ldl_cache(self, spd):
        """Alternating base values must invalidate the cached simplicial factor.

        Exercises the epoch-based rebuild of the persistent updown LDL' copy:
        each solve must reflect the base Ax it was called with, not a stale one.
        """
        Ai, Aj, Ax, A = spd
        rng = np.random.default_rng(13)
        n = A.shape[0]
        c = rng.normal(size=n)
        b = rng.normal(size=n)
        for s in (1.0, 2.5, 1.0, 0.7, 2.5):
            x = cholgraph.update_solve(Ai, Aj, s * Ax, c, b)
            np.testing.assert_allclose(
                x, np.linalg.solve(s * A + np.outer(c, c), b), rtol=1e-9
            )

    def test_downdate_indefinite_raises(self, spd):
        Ai, Aj, Ax, A = spd
        n = A.shape[0]
        c = np.full(n, 100.0)  # huge downdate -> not positive definite
        with pytest.raises(Exception, match="positive definite"):
            cholgraph.update_solve(
                Ai, Aj, Ax, c, np.ones(n), downdate=True
            ).block_until_ready()


class TestFactorSolve:
    def test_mean_and_logdet_from_one_factor(self, spd):
        Ai, Aj, Ax, A = spd
        n = A.shape[0]
        rng = np.random.default_rng(20)
        b = rng.normal(size=n)
        cholgraph.clear_cache()
        c0 = cholgraph.factorization_count()
        xs, ld = cholgraph.factor_solve(
            Ai, Aj, Ax, [(b, cholgraph.MODE_A)], want_logdet=True
        )
        jax.block_until_ready(xs)
        np.testing.assert_allclose(xs[0], np.linalg.solve(A, b), rtol=1e-10)
        np.testing.assert_allclose(float(ld), np.linalg.slogdet(A)[1], rtol=1e-10)
        # mean + logdet came from a single factorization
        assert cholgraph.factorization_count() - c0 == 1

    def test_multiple_chains_one_factor(self, spd):
        """mean, a factor-part chain, and logdet — all from ONE factorization."""
        Ai, Aj, Ax, A = spd
        n = A.shape[0]
        rng = np.random.default_rng(21)
        b = rng.normal(size=n)
        z = rng.normal(size=n)
        cholgraph.clear_cache()
        c0 = cholgraph.factorization_count()
        (mean, w), ld = cholgraph.factor_solve(
            Ai, Aj, Ax,
            [(b, cholgraph.MODE_A), (z, (cholgraph.MODE_LT, cholgraph.MODE_PT))],
            want_logdet=True,
        )
        jax.block_until_ready((mean, w, ld))
        assert cholgraph.factorization_count() - c0 == 1
        np.testing.assert_allclose(mean, np.linalg.solve(A, b), rtol=1e-10)
        np.testing.assert_allclose(float(ld), np.linalg.slogdet(A)[1], rtol=1e-10)
        # w = P' L^-T z, so A w should equal L L' applied appropriately; check
        # the defining property Cov: w has A^{-1} covariance <=> (L' P w) == z.
        # Simpler exact check: separate calls reproduce the same w.
        w_ref = cholgraph.solve(
            Ai, Aj, Ax,
            cholgraph.solve(Ai, Aj, Ax, z, mode=cholgraph.MODE_LT),
            mode=cholgraph.MODE_PT,
        )
        np.testing.assert_allclose(w, w_ref, rtol=1e-10)

    def test_matches_separate_calls(self, spd):
        Ai, Aj, Ax, A = spd
        n = A.shape[0]
        rng = np.random.default_rng(22)
        b = rng.normal(size=n)
        c = rng.normal(size=n)
        xs = cholgraph.factor_solve(
            Ai, Aj, Ax, [(b, cholgraph.MODE_A), (c, cholgraph.MODE_A)]
        )
        np.testing.assert_allclose(xs[0], cholgraph.solve(Ai, Aj, Ax, b), rtol=1e-12)
        np.testing.assert_allclose(xs[1], cholgraph.solve(Ai, Aj, Ax, c), rtol=1e-12)

    def test_logdet_only_empty_rhs(self, spd):
        Ai, Aj, Ax, A = spd
        (xs, ld) = cholgraph.factor_solve(
            Ai, Aj, Ax, [], want_logdet=True, n=A.shape[0]
        )
        assert xs == []
        np.testing.assert_allclose(float(ld), np.linalg.slogdet(A)[1], rtol=1e-10)

    def test_multi_rhs_entry(self, spd):
        Ai, Aj, Ax, A = spd
        n = A.shape[0]
        rng = np.random.default_rng(23)
        B = rng.normal(size=(n, 3))
        (X,) = cholgraph.factor_solve(Ai, Aj, Ax, [(B, cholgraph.MODE_A)])
        np.testing.assert_allclose(X, np.linalg.solve(A, B), rtol=1e-10)

    def test_under_jit(self, spd):
        Ai, Aj, Ax, A = spd
        n = A.shape[0]
        rng = np.random.default_rng(24)
        b = rng.normal(size=n)
        z = rng.normal(size=n)

        @jax.jit
        def f(Ax, b, z):
            (mean, w), ld = cholgraph.factor_solve(
                Ai, Aj, Ax,
                [(b, cholgraph.MODE_A),
                 (z, (cholgraph.MODE_LT, cholgraph.MODE_PT))],
                want_logdet=True,
            )
            return mean, w, ld

        mean, w, ld = f(Ax, b, z)
        np.testing.assert_allclose(mean, np.linalg.solve(A, b), rtol=1e-10)
        np.testing.assert_allclose(float(ld), np.linalg.slogdet(A)[1], rtol=1e-10)

    def test_vmap_same_A_one_factorization(self, spd):
        Ai, Aj, Ax, A = spd
        n = A.shape[0]
        rng = np.random.default_rng(25)
        bs = rng.normal(size=(6, n))
        cholgraph.clear_cache()
        c0 = cholgraph.factorization_count()
        xs = jax.vmap(
            lambda b: cholgraph.factor_solve(Ai, Aj, Ax, [(b, cholgraph.MODE_A)])[0]
        )(bs)
        jax.block_until_ready(xs)
        # identical A across the batch -> value cache collapses to 1 factorization
        assert cholgraph.factorization_count() - c0 == 1
        for i in range(6):
            np.testing.assert_allclose(xs[i], np.linalg.solve(A, bs[i]), rtol=1e-10)

    def test_vmap_different_A_one_factorization_each(self, spd):
        Ai, Aj, Ax, A = spd
        n = A.shape[0]
        rng = np.random.default_rng(26)
        scales = np.linspace(1.0, 2.0, 4)
        Axs = scales[:, None] * Ax
        zs = rng.normal(size=(4, n))
        b = rng.normal(size=n)
        cholgraph.clear_cache()
        c0 = cholgraph.factorization_count()
        f = jax.jit(
            jax.vmap(
                lambda Ax, z: cholgraph.sample_gaussian(Ai, Aj, Ax, b, z),
                in_axes=(0, 0),
            )
        )
        etas, means = f(Axs, zs)
        jax.block_until_ready((etas, means))
        # 4 elements, each needs mean+sample+... but ONE factorization apiece
        assert cholgraph.factorization_count() - c0 == 4
        for i, s in enumerate(scales):
            np.testing.assert_allclose(
                means[i], np.linalg.solve(s * A, b), rtol=1e-10
            )

    def test_vmap_lowers_to_single_batched_call(self, spd):
        Ai, Aj, Ax, A = spd
        n = A.shape[0]
        bs = np.ones((4, n))
        f = jax.jit(
            jax.vmap(
                lambda b: cholgraph.factor_solve(Ai, Aj, Ax, [(b, cholgraph.MODE_A)])[0]
            )
        )
        hlo = f.lower(bs).compile().as_text()
        assert hlo.count("cholgraph_factor_solve_batched_f64") == 1

    def test_vmap_over_pattern_raises(self, spd):
        Ai, Aj, Ax, A = spd
        n = A.shape[0]
        Ais = np.broadcast_to(Ai, (3,) + Ai.shape)
        b = np.ones(n)
        with pytest.raises(ValueError, match="sparsity pattern"):
            jax.vmap(
                lambda Ai: cholgraph.factor_solve(Ai, Aj, Ax, [(b, cholgraph.MODE_A)])[0]
            )(Ais)


class TestSampleGaussian:
    def test_mean_and_covariance(self, spd):
        Ai, Aj, Ax, A = spd
        n = A.shape[0]
        rng = np.random.default_rng(27)
        b = rng.normal(size=n)
        eta, mean = cholgraph.sample_gaussian(Ai, Aj, Ax, b, rng.normal(size=n))
        np.testing.assert_allclose(mean, np.linalg.solve(A, b), rtol=1e-10)
        # empirical covariance of many draws ~ A^{-1}
        zs = rng.normal(size=(6000, n))
        etas = jax.vmap(
            lambda z: cholgraph.sample_gaussian(Ai, Aj, Ax, b, z)[0]
        )(zs)
        emp = np.cov(np.asarray(etas).T)
        np.testing.assert_allclose(emp, np.linalg.inv(A), atol=0.05)

    def test_logdet_option(self, spd):
        Ai, Aj, Ax, A = spd
        n = A.shape[0]
        rng = np.random.default_rng(28)
        b = rng.normal(size=n)
        z = rng.normal(size=n)
        eta, mean, ld = cholgraph.sample_gaussian(
            Ai, Aj, Ax, b, z, want_logdet=True
        )
        np.testing.assert_allclose(float(ld), np.linalg.slogdet(A)[1], rtol=1e-10)
        np.testing.assert_allclose(mean, np.linalg.solve(A, b), rtol=1e-10)


class TestBCOO:
    def _bcoo(self, A):
        from jax.experimental import sparse as jsparse

        return jsparse.BCOO.fromdense(jnp.asarray(A))

    def test_solve_bcoo(self, spd):
        Ai, Aj, Ax, A = spd
        M = self._bcoo(A)
        b = np.arange(1.0, A.shape[0] + 1.0)
        np.testing.assert_allclose(
            cholgraph.solve_bcoo(M, b), np.linalg.solve(A, b), rtol=1e-10
        )

    def test_solve_bcoo_jit_caches(self, spd):
        Ai, Aj, Ax, A = spd
        M = self._bcoo(A)
        b = np.ones(A.shape[0])
        cholgraph.clear_cache()
        f = jax.jit(cholgraph.solve_bcoo)
        f(M, b).block_until_ready()
        f(M, 2.0 * b).block_until_ready()
        assert cholgraph.cache_size() == 1  # same pattern reused

    def test_logdet_bcoo(self, spd):
        Ai, Aj, Ax, A = spd
        M = self._bcoo(A)
        np.testing.assert_allclose(
            float(cholgraph.logdet_bcoo(M)),
            np.linalg.slogdet(A)[1],
            rtol=1e-10,
        )

    def test_update_solve_bcoo(self, spd):
        Ai, Aj, Ax, A = spd
        M = self._bcoo(A)
        rng = np.random.default_rng(12)
        n = A.shape[0]
        c = rng.normal(size=n)
        b = rng.normal(size=n)
        np.testing.assert_allclose(
            cholgraph.update_solve_bcoo(M, c, b),
            np.linalg.solve(A + np.outer(c, c), b),
            rtol=1e-9,
        )

    def test_rejects_batched_bcoo(self, spd):
        from jax.experimental import sparse as jsparse

        Ai, Aj, Ax, A = spd
        stacked = jsparse.BCOO.fromdense(jnp.stack([jnp.asarray(A)] * 2), n_batch=1)
        with pytest.raises(ValueError, match="n_batch"):
            cholgraph.solve_bcoo(stacked, np.ones(A.shape[0]))


class TestAD:
    def test_grad_matches_finite_differences(self, spd):
        Ai, Aj, Ax, A = spd
        n = A.shape[0]
        b = np.linspace(1.0, 2.0, n)
        w = np.linspace(-1.0, 1.0, n)  # nontrivial loss weights

        def loss(Ax, b):
            return (w * cholgraph.solve(Ai, Aj, Ax, b)).sum()

        gAx, gb = jax.grad(loss, argnums=(0, 1))(jnp.asarray(Ax), jnp.asarray(b))

        def dense_loss(Axv, bv):
            Ad = np.zeros_like(A)
            for k in range(len(Ai)):
                if Ai[k] <= Aj[k]:
                    Ad[Ai[k], Aj[k]] = Axv[k]
                    Ad[Aj[k], Ai[k]] = Axv[k]
            return (w * np.linalg.solve(Ad, bv)).sum()

        eps = 1e-6
        rng = np.random.default_rng(3)
        for k in rng.choice(len(Ax), size=10, replace=False):
            e = np.zeros_like(Ax)
            e[k] = eps
            fd = (dense_loss(Ax + e, b) - dense_loss(Ax - e, b)) / (2 * eps)
            np.testing.assert_allclose(gAx[k], fd, atol=1e-5)
        for k in rng.choice(n, size=5, replace=False):
            e = np.zeros_like(b)
            e[k] = eps
            fd = (dense_loss(Ax, b + e) - dense_loss(Ax, b - e)) / (2 * eps)
            np.testing.assert_allclose(gb[k], fd, atol=1e-5)

    def test_grad_under_jit(self, spd):
        Ai, Aj, Ax, A = spd
        b = np.ones(A.shape[0])
        g1 = jax.grad(lambda Ax: cholgraph.solve(Ai, Aj, Ax, b).sum())(jnp.asarray(Ax))
        g2 = jax.jit(jax.grad(lambda Ax: cholgraph.solve(Ai, Aj, Ax, b).sum()))(
            jnp.asarray(Ax)
        )
        np.testing.assert_allclose(g1, g2, rtol=1e-12)


class TestLogdet:
    def test_logdet_matches_slogdet(self, spd):
        Ai, Aj, Ax, A = spd
        ld = cholgraph.logdet(Ai, Aj, Ax, A.shape[0])
        np.testing.assert_allclose(float(ld), np.linalg.slogdet(A)[1], rtol=1e-10)

    def test_logdet_under_jit(self, spd):
        Ai, Aj, Ax, A = spd
        f = jax.jit(lambda Ax: cholgraph.logdet(Ai, Aj, Ax, A.shape[0]))
        for s in (1.0, 2.0):
            np.testing.assert_allclose(
                float(f(s * Ax)), np.linalg.slogdet(s * A)[1], rtol=1e-10
            )


class TestSelinv:
    def test_matches_dense_inverse_at_pattern(self, spd):
        Ai, Aj, Ax, A = spd
        n = A.shape[0]
        z = np.asarray(cholgraph.selinv(Ai, Aj, Ax, n))
        Ainv = np.linalg.inv(A)
        # Selected inverse must match the dense inverse at every COO position,
        # for both upper- and lower-triangle entries (symmetry).
        np.testing.assert_allclose(z, Ainv[Ai, Aj], atol=1e-10)

    def test_diagonal_is_marginal_variance(self, spd):
        Ai, Aj, Ax, A = spd
        n = A.shape[0]
        z = np.asarray(cholgraph.selinv(Ai, Aj, Ax, n))
        d = Ai == Aj
        # diag(A^-1) read off the diagonal COO entries.
        np.testing.assert_allclose(
            z[d][np.argsort(Ai[d])], np.diag(np.linalg.inv(A)), atol=1e-10
        )

    def test_upper_only_input(self, spd):
        Ai, Aj, Ax, A = spd
        keep = Ai <= Aj
        n = A.shape[0]
        z = np.asarray(cholgraph.selinv(Ai[keep], Aj[keep], Ax[keep], n))
        np.testing.assert_allclose(z, np.linalg.inv(A)[Ai[keep], Aj[keep]], atol=1e-10)

    def test_under_jit(self, spd):
        Ai, Aj, Ax, A = spd
        n = A.shape[0]
        z = jax.jit(lambda Ax: cholgraph.selinv(Ai, Aj, Ax, n))(jnp.asarray(Ax))
        np.testing.assert_allclose(np.asarray(z), np.linalg.inv(A)[Ai, Aj], atol=1e-10)


class TestLogdetGrad:
    def test_grad_matches_finite_differences(self, spd):
        Ai, Aj, Ax, A = spd
        n = A.shape[0]
        g = np.asarray(jax.grad(lambda x: cholgraph.logdet(Ai, Aj, x, n))(jnp.asarray(Ax)))

        def dense_logdet(Axv):
            Ad = np.zeros_like(A)
            for k in range(len(Ai)):
                if Ai[k] <= Aj[k]:
                    Ad[Ai[k], Aj[k]] = Axv[k]
                    Ad[Aj[k], Ai[k]] = Axv[k]
            return np.linalg.slogdet(Ad)[1]

        eps = 1e-6
        rng = np.random.default_rng(5)
        for k in rng.choice(len(Ax), size=12, replace=False):
            e = np.zeros_like(Ax)
            e[k] = eps
            fd = (dense_logdet(Ax + e) - dense_logdet(Ax - e)) / (2 * eps)
            np.testing.assert_allclose(g[k], fd, atol=1e-5)

    def test_grad_under_jit(self, spd):
        Ai, Aj, Ax, A = spd
        n = A.shape[0]

        def f(x):
            return cholgraph.logdet(Ai, Aj, x, n)

        g1 = jax.grad(f)(jnp.asarray(Ax))
        g2 = jax.jit(jax.grad(f))(jnp.asarray(Ax))
        np.testing.assert_allclose(g1, g2, rtol=1e-12)

    def test_vmap_grad(self, spd):
        Ai, Aj, Ax, A = spd
        n = A.shape[0]
        Axb = jnp.stack([jnp.asarray(Ax) * (1.0 + 0.05 * i) for i in range(4)])

        def gf(x):
            return jax.grad(lambda a: cholgraph.logdet(Ai, Aj, a, n))(x)

        gb = jax.vmap(gf)(Axb)
        gref = jnp.stack([gf(Axb[i]) for i in range(4)])
        np.testing.assert_allclose(np.asarray(gb), np.asarray(gref), rtol=1e-10)

    def test_gaussian_log_posterior_gradient(self, spd):
        """The NUTS use case: grad of -1/2 b'A^-1 b + 1/2 log|A| combines the
        solve VJP (quadratic term) and the logdet VJP (selected inverse)."""
        Ai, Aj, Ax, A = spd
        n = A.shape[0]
        b = jnp.asarray(np.linspace(1.0, 2.0, n))

        def neg_log_post(x):
            quad = b @ cholgraph.solve(Ai, Aj, x, b)
            return 0.5 * quad - 0.5 * cholgraph.logdet(Ai, Aj, x, n)

        g = np.asarray(jax.grad(neg_log_post)(jnp.asarray(Ax)))

        def dense(Axv):
            Ad = np.zeros_like(A)
            for k in range(len(Ai)):
                if Ai[k] <= Aj[k]:
                    Ad[Ai[k], Aj[k]] = Axv[k]
                    Ad[Aj[k], Ai[k]] = Axv[k]
            bn = np.asarray(b)
            return 0.5 * bn @ np.linalg.solve(Ad, bn) - 0.5 * np.linalg.slogdet(Ad)[1]

        eps = 1e-6
        rng = np.random.default_rng(7)
        for k in rng.choice(len(Ax), size=10, replace=False):
            e = np.zeros_like(Ax)
            e[k] = eps
            fd = (dense(Ax + e) - dense(Ax - e)) / (2 * eps)
            np.testing.assert_allclose(g[k], fd, atol=1e-5)


class TestSolveModes:
    def test_factor_chain_equals_full_solve(self, spd):
        """P' L^-T L^-1 P b must equal A^-1 b."""
        Ai, Aj, Ax, A = spd
        b = np.arange(1.0, A.shape[0] + 1.0)

        def s(v, m):
            return cholgraph.solve(Ai, Aj, Ax, v, mode=m)

        chained = s(
            s(s(s(b, cholgraph.MODE_P), cholgraph.MODE_L), cholgraph.MODE_LT),
            cholgraph.MODE_PT,
        )
        np.testing.assert_allclose(chained, np.linalg.solve(A, b), rtol=1e-10)


class TestErrors:
    def test_not_positive_definite(self):
        Ai = np.array([0, 1], dtype=np.int32)
        Aj = np.array([0, 1], dtype=np.int32)
        Ax = np.array([1.0, -1.0])  # indefinite
        with pytest.raises(Exception, match="positive definite"):
            cholgraph.solve(Ai, Aj, Ax, np.array([1.0, 1.0])).block_until_ready()

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="equal lengths"):
            cholgraph.solve(
                np.array([0], dtype=np.int32),
                np.array([0, 1], dtype=np.int32),
                np.array([1.0]),
                np.array([1.0]),
            )

    def test_index_out_of_range(self):
        Ai = np.array([0, 5], dtype=np.int32)
        Aj = np.array([0, 5], dtype=np.int32)
        Ax = np.array([1.0, 1.0])
        with pytest.raises(Exception, match="out of range"):
            cholgraph.solve(Ai, Aj, Ax, np.array([1.0, 1.0])).block_until_ready()


class TestCache:
    def test_cache_grows_and_clears(self, spd):
        cholgraph.clear_cache()
        assert cholgraph.cache_size() == 0
        Ai, Aj, Ax, A = spd
        b = np.ones(A.shape[0])
        cholgraph.solve(Ai, Aj, Ax, b).block_until_ready()
        assert cholgraph.cache_size() == 1
        cholgraph.solve(Ai, Aj, 2.0 * Ax, b).block_until_ready()
        assert cholgraph.cache_size() == 1  # same pattern, no new entry
        Bi, Bj, Bx, B = grid_laplacian(5)
        cholgraph.solve(Bi, Bj, Bx, np.ones(B.shape[0])).block_until_ready()
        assert cholgraph.cache_size() == 2
        cholgraph.clear_cache()
        assert cholgraph.cache_size() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
