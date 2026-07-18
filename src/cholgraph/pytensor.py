"""PyTensor frontend for cholgraph (optional; ``pip install cholgraph[pytensor]``).

A second frontend over the same cached CHOLMOD core as the JAX API, for use with
PyMC's default (PyTensor) backend — including gradient-based samplers like NUTS.
These are pure-Python :class:`pytensor.graph.op.Op` s whose ``perform`` calls the
numpy-callable core (``cholgraph_cpp.solve_np`` / ``logdet_np`` / ``selinv_np``);
they do **not** go through JAX/XLA, so they run under PyTensor's ordinary C/numba
backend.

``solve`` and ``logdet`` carry the same reverse-mode rules as their JAX
counterparts, so a Gaussian log-density built from them is differentiable in the
matrix values ``Ax`` (the selected inverse :func:`selinv` supplies ``logdet``'s
gradient). Example::

    import pytensor.tensor as pt
    import cholgraph.pytensor as cjpt

    Ax = pt.dvector("Ax")
    # -1/2 b' A^-1 b + 1/2 log|A|  (up to constants); grad flows into Ax
    quad = b @ cjpt.solve(Ai, Aj, Ax, b)
    logp = -0.5 * quad + 0.5 * cjpt.logdet(Ai, Aj, Ax, n)
    g = pt.grad(logp, Ax)

The sparsity pattern ``(Ai, Aj)`` is data (integer, non-differentiable); only
``Ax`` and ``b`` carry gradients.
"""

import numpy as np

try:
    import pytensor.tensor as pt
    from pytensor.graph.basic import Apply
    from pytensor.graph.op import Op
    from pytensor.gradient import grad_undefined
except ImportError as exc:  # pragma: no cover - exercised only without pytensor
    raise ImportError(
        "cholgraph.pytensor requires PyTensor. Install it with "
        "'pip install cholgraph[pytensor]' (or 'pip install pytensor')."
    ) from exc

import cholgraph_cpp as _cpp

from . import MODE_A

__all__ = ["solve", "logdet", "selinv", "CholmodSolve", "CholmodLogdet", "CholmodSelinv"]


def _coo(Ai, Aj):
    return pt.cast(pt.as_tensor_variable(Ai), "int32"), pt.cast(
        pt.as_tensor_variable(Aj), "int32"
    )


class CholmodSolve(Op):
    """Solve ``A x = b`` (or a factor-part system) from cached CHOLMOD factor.

    Differentiable in ``Ax`` and ``b`` only for ``mode == MODE_A``; other modes
    are forward-only (their VJP is undefined, matching the JAX ``solve``).
    """

    __props__ = ("mode",)

    def __init__(self, mode=MODE_A):
        self.mode = int(mode)

    def make_node(self, Ai, Aj, Ax, b):
        Ai, Aj = _coo(Ai, Aj)
        Ax = pt.cast(pt.as_tensor_variable(Ax), "float64")
        b = pt.cast(pt.as_tensor_variable(b), "float64")
        if b.ndim not in (1, 2):
            raise ValueError("b must be 1D or 2D")
        return Apply(self, [Ai, Aj, Ax, b], [b.type()])

    def perform(self, node, inputs, outputs):
        Ai, Aj, Ax, b = inputs
        outputs[0][0] = _cpp.solve_np(
            np.ascontiguousarray(Ai, np.int32),
            np.ascontiguousarray(Aj, np.int32),
            np.ascontiguousarray(Ax, np.float64),
            np.ascontiguousarray(b, np.float64),
            self.mode,
        )

    def infer_shape(self, fgraph, node, input_shapes):
        return [input_shapes[3]]

    # Reverse-mode rule. PyTensor >= 3.1 calls `pullback`; older versions call
    # `grad` (see Op.L_op dispatch). Both delegate to the same VJP, which does
    # not need `outputs`, so the two wrappers share one implementation.
    def pullback(self, inputs, outputs, cotangents):
        return self._vjp(inputs, cotangents)

    def grad(self, inputs, output_grads):
        return self._vjp(inputs, output_grads)

    def _vjp(self, inputs, cotangents):
        Ai, Aj, Ax, b = inputs
        (g,) = cotangents
        if self.mode != MODE_A:
            # Factor-part solves are building blocks; AD through them is not
            # defined (same policy as the JAX solve).
            return [
                grad_undefined(self, 0, Ai),
                grad_undefined(self, 1, Aj),
                grad_undefined(self, 2, Ax),
                grad_undefined(self, 3, b),
            ]
        # x = A^-1 b, v = A^-1 g. For the stored upper-triangle value at (i, j),
        # dAx = -(v_i x_j + v_j x_i) off-diagonal, -v_i x_i on the diagonal;
        # ignored lower-triangle entries get zero. db = v.
        x = self(Ai, Aj, Ax, b)
        v = _solve_a(Ai, Aj, Ax, g)
        if b.ndim == 1:
            cross = v[Ai] * x[Aj] + v[Aj] * x[Ai]
            diag = v[Ai] * x[Ai]
        else:
            cross = (v[Ai] * x[Aj]).sum(-1) + (v[Aj] * x[Ai]).sum(-1)
            diag = (v[Ai] * x[Ai]).sum(-1)
        dAx = -pt.where(pt.eq(Ai, Aj), diag, pt.where(pt.lt(Ai, Aj), cross, 0.0))
        return [grad_undefined(self, 0, Ai), grad_undefined(self, 1, Aj), dAx, v]


_solve_a = CholmodSolve(MODE_A)


class CholmodSelinv(Op):
    """Selected inverse: ``A^-1`` at ``A``'s pattern. Forward-only."""

    __props__ = ("n",)

    def __init__(self, n):
        self.n = int(n)

    def make_node(self, Ai, Aj, Ax):
        Ai, Aj = _coo(Ai, Aj)
        Ax = pt.cast(pt.as_tensor_variable(Ax), "float64")
        return Apply(self, [Ai, Aj, Ax], [Ax.type()])

    def perform(self, node, inputs, outputs):
        Ai, Aj, Ax = inputs
        outputs[0][0] = _cpp.selinv_np(
            np.ascontiguousarray(Ai, np.int32),
            np.ascontiguousarray(Aj, np.int32),
            np.ascontiguousarray(Ax, np.float64),
            self.n,
        )

    def infer_shape(self, fgraph, node, input_shapes):
        return [input_shapes[2]]


class CholmodLogdet(Op):
    """``log|A|``. Differentiable in ``Ax`` via the selected inverse."""

    __props__ = ("n",)

    def __init__(self, n):
        self.n = int(n)

    def make_node(self, Ai, Aj, Ax):
        Ai, Aj = _coo(Ai, Aj)
        Ax = pt.cast(pt.as_tensor_variable(Ax), "float64")
        return Apply(self, [Ai, Aj, Ax], [pt.scalar(dtype="float64")])

    def perform(self, node, inputs, outputs):
        Ai, Aj, Ax = inputs
        val = _cpp.logdet_np(
            np.ascontiguousarray(Ai, np.int32),
            np.ascontiguousarray(Aj, np.int32),
            np.ascontiguousarray(Ax, np.float64),
            self.n,
        )
        outputs[0][0] = np.asarray(val, dtype=np.float64)

    def pullback(self, inputs, outputs, cotangents):
        return self._vjp(inputs, cotangents)

    def grad(self, inputs, output_grads):
        return self._vjp(inputs, output_grads)

    def _vjp(self, inputs, cotangents):
        Ai, Aj, Ax = inputs
        (g,) = cotangents
        # d log|A| / dA = A^-1; a stored upper-triangle value at (i, j) drives
        # both A_ij and A_ji, so its sensitivity is 2 (A^-1)_ij; a diagonal
        # value contributes (A^-1)_ii once; ignored lower entries get zero.
        z = CholmodSelinv(self.n)(Ai, Aj, Ax)
        dAx = g * pt.where(pt.eq(Ai, Aj), z, pt.where(pt.lt(Ai, Aj), 2.0 * z, 0.0))
        return [grad_undefined(self, 0, Ai), grad_undefined(self, 1, Aj), dAx]


def solve(Ai, Aj, Ax, b, mode=MODE_A):
    """Solve ``A x = b`` for SPD sparse ``A`` (PyTensor). See :func:`cholgraph.solve`.

    ``Ai, Aj`` are the COO pattern (int, non-differentiable); ``Ax`` the values
    and ``b`` the right-hand side(s) carry gradients when ``mode == MODE_A``.
    """
    return CholmodSolve(mode)(Ai, Aj, Ax, b)


def logdet(Ai, Aj, Ax, n):
    """``log|A|`` (PyTensor), differentiable in ``Ax``. See :func:`cholgraph.logdet`."""
    return CholmodLogdet(int(n))(Ai, Aj, Ax)


def selinv(Ai, Aj, Ax, n):
    """Selected inverse ``A^-1`` at the pattern (PyTensor). See :func:`cholgraph.selinv`."""
    return CholmodSelinv(int(n))(Ai, Aj, Ax)
