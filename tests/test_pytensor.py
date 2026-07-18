"""Tests for the optional PyTensor frontend (cholgraph.pytensor).

Skipped entirely if PyTensor is not installed.
"""

import numpy as np
import pytest

pytensor = pytest.importorskip("pytensor")
import pytensor.tensor as pt  # noqa: E402
from pytensor.gradient import verify_grad  # noqa: E402

import cholgraph  # noqa: E402
import cholgraph.pytensor as cjpt  # noqa: E402


def spd_coo(n=16, seed=0):
    """Full-symmetric SPD COO plus its dense matrix."""
    r = np.random.default_rng(seed)
    M = r.standard_normal((n, n)) * (r.random((n, n)) < 0.3)
    A = M @ M.T + n * np.eye(n)
    Ai, Aj = np.nonzero(A)
    return (
        Ai.astype(np.int32),
        Aj.astype(np.int32),
        np.ascontiguousarray(A[Ai, Aj]),
        A,
    )


@pytest.fixture
def problem():
    Ai, Aj, Ax, A = spd_coo()
    n = A.shape[0]
    b = np.ascontiguousarray(np.linspace(1.0, 2.0, n))
    return Ai, Aj, Ax, A, n, b


class TestForward:
    def test_solve(self, problem):
        Ai, Aj, Ax, A, n, b = problem
        Axv, bv = pt.dvector("Ax"), pt.dvector("b")
        f = pytensor.function([Axv, bv], cjpt.solve(Ai, Aj, Axv, bv))
        np.testing.assert_allclose(f(Ax, b), np.linalg.solve(A, b), rtol=1e-10)

    def test_solve_multi_rhs(self, problem):
        Ai, Aj, Ax, A, n, b = problem
        B = np.ascontiguousarray(np.random.default_rng(1).standard_normal((n, 4)))
        Axv, Bv = pt.dvector("Ax"), pt.dmatrix("B")
        f = pytensor.function([Axv, Bv], cjpt.solve(Ai, Aj, Axv, Bv))
        np.testing.assert_allclose(f(Ax, B), np.linalg.solve(A, B), rtol=1e-10)

    def test_logdet(self, problem):
        Ai, Aj, Ax, A, n, b = problem
        Axv = pt.dvector("Ax")
        f = pytensor.function([Axv], cjpt.logdet(Ai, Aj, Axv, n))
        np.testing.assert_allclose(float(f(Ax)), np.linalg.slogdet(A)[1], rtol=1e-10)

    def test_selinv(self, problem):
        Ai, Aj, Ax, A, n, b = problem
        Axv = pt.dvector("Ax")
        f = pytensor.function([Axv], cjpt.selinv(Ai, Aj, Axv, n))
        np.testing.assert_allclose(f(Ax), np.linalg.inv(A)[Ai, Aj], atol=1e-10)

    def test_matches_jax_frontend(self, problem):
        # The PyTensor and JAX frontends share the core; results must agree.
        Ai, Aj, Ax, A, n, b = problem
        Axv, bv = pt.dvector("Ax"), pt.dvector("b")
        x_pt = pytensor.function([Axv, bv], cjpt.solve(Ai, Aj, Axv, bv))(Ax, b)
        ld_pt = float(pytensor.function([Axv], cjpt.logdet(Ai, Aj, Axv, n))(Ax))
        np.testing.assert_allclose(x_pt, cholgraph.solve(Ai, Aj, Ax, b), rtol=1e-12)
        np.testing.assert_allclose(
            ld_pt, float(cholgraph.logdet(Ai, Aj, Ax, n)), rtol=1e-12
        )


class TestGrad:
    def test_verify_grad_solve(self, problem):
        Ai, Aj, Ax, A, n, b = problem
        rng = np.random.default_rng(0)
        verify_grad(
            lambda ax: cjpt.solve(Ai, Aj, ax, b),
            [Ax],
            rng=rng,
            abs_tol=1e-4,
            rel_tol=1e-4,
        )

    def test_verify_grad_logdet(self, problem):
        Ai, Aj, Ax, A, n, b = problem
        rng = np.random.default_rng(0)
        verify_grad(
            lambda ax: cjpt.logdet(Ai, Aj, ax, n),
            [Ax],
            rng=rng,
            abs_tol=1e-4,
            rel_tol=1e-4,
        )

    def test_grad_wrt_b(self, problem):
        # d/db [w' A^-1 b] = A^-1 w (A symmetric).
        Ai, Aj, Ax, A, n, b = problem
        w = np.ascontiguousarray(np.linspace(-1.0, 1.0, n))
        Axv, bv = pt.dvector("Ax"), pt.dvector("b")
        loss = pt.dot(w, cjpt.solve(Ai, Aj, Axv, bv))
        gb = pytensor.function([Axv, bv], pt.grad(loss, bv))(Ax, b)
        np.testing.assert_allclose(gb, np.linalg.solve(A, w), rtol=1e-9)

    def test_gaussian_log_posterior_gradient(self, problem):
        """The NUTS use case: grad of 1/2 b'A^-1 b - 1/2 log|A| in Ax, combining
        the solve VJP and the logdet (selected-inverse) VJP."""
        Ai, Aj, Ax, A, n, b = problem
        Axv, bv = pt.dvector("Ax"), pt.dvector("b")
        logp = 0.5 * pt.dot(bv, cjpt.solve(Ai, Aj, Axv, bv)) - 0.5 * cjpt.logdet(
            Ai, Aj, Axv, n
        )
        g = np.asarray(pytensor.function([Axv, bv], pt.grad(logp, Axv))(Ax, b))

        def dense(Axv_):
            # Reconstruct exactly as the core does: read the upper triangle and
            # mirror it (lower-triangle COO entries are ignored).
            Ad = np.zeros_like(A)
            for k in range(len(Ai)):
                if Ai[k] <= Aj[k]:
                    Ad[Ai[k], Aj[k]] = Axv_[k]
                    Ad[Aj[k], Ai[k]] = Axv_[k]
            bn = np.asarray(b)
            return 0.5 * bn @ np.linalg.solve(Ad, bn) - 0.5 * np.linalg.slogdet(Ad)[1]

        eps = 1e-6
        rng = np.random.default_rng(2)
        for k in rng.choice(len(Ax), size=12, replace=False):
            e = np.zeros_like(Ax)
            e[k] = eps
            fd = (dense(Ax + e) - dense(Ax - e)) / (2 * eps)
            np.testing.assert_allclose(g[k], fd, atol=1e-5)

    def test_grad_uses_pullback_no_futurewarning(self, problem):
        # PyTensor >= 3.1 must route through Op.pullback (not the deprecated
        # grad/L_op path), so building the gradient raises no FutureWarning.
        Ai, Aj, Ax, A, n, b = problem
        Axv, bv = pt.dvector("Ax"), pt.dvector("b")
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            logp = 0.5 * pt.dot(bv, cjpt.solve(Ai, Aj, Axv, bv)) - 0.5 * cjpt.logdet(
                Ai, Aj, Axv, n
            )
            pt.grad(logp, Axv)  # would raise if the deprecated path were used
