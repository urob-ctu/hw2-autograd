#!/usr/bin/env python
"""Local tests of HW2: python test.py [-v]

Every check compares your Tensor with plain numpy: the value of an expression,
and the gradients that backward() leaves in the inputs with numerical gradients
(central differences). Functions you have not implemented yet simply fail, so
you can run this from the very first line you write.

The grading runs the same kind of checks with other expressions, shapes and
values - add your own cases at the bottom of this file.
"""

import sys
import traceback

import numpy as np

failures = 0
rng = np.random.default_rng(42)


def rnd(*shape):
    """Random values of both signs that keep 0.1 away from zero (relu kink, division)."""
    x = rng.normal(scale=2.0, size=shape)
    return np.sign(x) * (np.abs(x) + 0.1)


def pos(*shape):
    return rng.uniform(0.5, 2.0, size=shape)


def error_text():
    """The whole traceback with `python test.py -v`, otherwise only its last line."""
    text = traceback.format_exc().strip()
    return text.replace("\n", "\n        ") if "-v" in sys.argv else text.splitlines()[-1] + "   (-v shows the traceback)"


def report(name, problem):
    global failures
    failures += problem is not None
    print(f"{'PASSED' if problem is None else 'FAILED'}  {name}" + (f"\n        {problem}" if problem else ""))


def difference(label, got, want, rtol, atol):
    got, want = np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64)
    if got.shape != want.shape:
        return f"{label} has shape {got.shape}, expected {want.shape}"
    if not np.allclose(got, want, rtol=rtol, atol=atol):
        worst = np.unravel_index(np.argmax(np.abs(got - want)), want.shape) if want.ndim else ()
        return f"{label} is wrong: got {got[worst]:.6g}, expected {want[worst]:.6g} at index {worst}"
    return None


def numerical_gradient(f, inputs, i, eps=1e-6):
    """d sum(f(*inputs)) / d inputs[i]"""
    xs = [np.array(x, dtype=np.float64) for x in inputs]
    grad = np.zeros_like(xs[i])
    for idx in np.ndindex(grad.shape):
        orig = xs[i][idx]
        xs[i][idx] = orig + eps
        hi = np.sum(f(*xs))
        xs[i][idx] = orig - eps
        lo = np.sum(f(*xs))
        xs[i][idx] = orig
        grad[idx] = (hi - lo) / (2 * eps)
    return grad


def check(name, tensor_fn, numpy_fn, *inputs):
    """tensor_fn gets Tensors, numpy_fn gets the same values as numpy arrays."""
    try:
        tensors = [Tensor(np.copy(x)) for x in inputs]
        out = tensor_fn(*tensors)
        problem = difference("the result", out.data, numpy_fn(*inputs), 1e-6, 1e-9)
        if problem is None:
            out.backward()  # the gradient of the output is set to ones, i.e. we differentiate sum(out)
            for i, t in enumerate(tensors):
                problem = problem or difference(f"the gradient of input {i}", t.grad, numerical_gradient(numpy_fn, inputs, i), 1e-4, 1e-6)
    except Exception:
        problem = error_text()
    report(name, problem)


def check_values(name, procedure, expected):
    """procedure() returns a list of arrays that has to match `expected`."""
    try:
        problem = None
        for i, (got, want) in enumerate(zip(procedure(), expected)):
            problem = problem or difference(f"value {i}", got, want, 1e-6, 1e-9)
    except Exception:
        problem = error_text()
    report(name, problem)


def numpy_cross_entropy(logits, target):
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_softmax = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return -log_softmax[np.arange(len(target)), target].mean()


if __name__ == "__main__":
    try:
        from engine import Tensor, reshape_gradient
    except Exception:
        print(f"engine.py could not be imported:\n{traceback.format_exc()}")
        sys.exit(1)

    print("--- reshape_gradient (everything that broadcasts depends on it, get it right first)")
    G, G3 = rnd(2, 3), rnd(2, 3, 4)
    check_values("nothing was broadcast: (2, 3) stays (2, 3)", lambda: [reshape_gradient(G, (2, 3))], [G])
    check_values("scalar target: (2, 3) -> ()", lambda: [reshape_gradient(G, ())], [G.sum()])
    check_values("target of lower rank: (2, 3) -> (3,)", lambda: [reshape_gradient(G, (3,))], [G.sum(axis=0)])
    check_values("stretched axis of size 1: (2, 3) -> (2, 1)", lambda: [reshape_gradient(G, (2, 1))], [G.sum(axis=1, keepdims=True)])
    check_values("both at once: (2, 3, 4) -> (3, 1)", lambda: [reshape_gradient(G3, (3, 1))], [G3.sum(axis=(0, 2)).reshape(3, 1)])

    print("--- basic operations")
    check("a + b", lambda a, b: a + b, lambda a, b: a + b, rnd(2, 3), rnd(2, 3))
    check("a + b, broadcasting (3,) to (2, 3)", lambda a, b: a + b, lambda a, b: a + b, rnd(3), rnd(2, 3))
    check("a * b, broadcasting (2, 1) to (2, 3)", lambda a, b: a * b, lambda a, b: a * b, rnd(2, 1), rnd(2, 3))
    check("a - b, scalar and vector", lambda a, b: a - b, lambda a, b: a - b, rnd(), rnd(4))
    check("a / b", lambda a, b: a / b, lambda a, b: a / b, rnd(2, 3), rnd(2, 3))
    check("a ** 3", lambda a: a**3, lambda a: a**3, rnd(2, 3))
    check("1 - 2 * a", lambda a: 1 - 2 * a, lambda a: 1 - 2 * a, rnd(2, 3))

    print("--- functions")
    check("sin", lambda a: a.sin(), np.sin, rnd(2, 3))
    check("cos", lambda a: a.cos(), np.cos, rnd(2, 3))
    check("exp", lambda a: a.exp(), np.exp, rnd(2, 3))
    check("log", lambda a: a.log(), np.log, pos(2, 3))
    check("relu", lambda a: a.relu(), lambda a: np.maximum(a, 0), rnd(2, 3))
    check("sigmoid", lambda a: a.sigmoid(), lambda a: 1 / (1 + np.exp(-a)), rnd(2, 3))
    check("tanh", lambda a: a.tanh(), np.tanh, rnd(2, 3))

    print("--- sum, matmul, graphs")
    check("sum()", lambda a: a.sum(), np.sum, rnd(2, 3))
    check("sum(axis=1) * w", lambda a, w: a.sum(axis=1) * w, lambda a, w: a.sum(axis=1) * w, rnd(2, 3), rnd(2))
    check("a @ b", lambda a, b: a @ b, lambda a, b: a @ b, rnd(4, 3), rnd(3, 2))
    check("a @ b with a batch dimension", lambda a, b: a @ b, lambda a, b: a @ b, rnd(5, 4, 3), rnd(3, 2))
    check("a * a + a", lambda a: a * a + a, lambda a: a * a + a, rnd(2, 3))
    check("cos(a + b) / (sin(a) + 2) + sigmoid(a * b)", lambda a, b: (a + b).cos() / (a.sin() + 2) + (a * b).sigmoid(),
          lambda a, b: np.cos(a + b) / (np.sin(a) + 2) + 1 / (1 + np.exp(-a * b)), rnd(3), rnd(2, 3))  # fmt: skip

    print("--- losses")
    target = np.array([2, 0, 1, 2])
    check("regularization_loss", lambda a: a.regularization_loss(0.1), lambda a: 0.1 * np.sum(a**2), rnd(2, 3))
    check("cross_entropy_loss", lambda a: a.cross_entropy_loss(target), lambda a: numpy_cross_entropy(a, target), rnd(4, 3))
    check("cross_entropy_loss * 2", lambda a: a.cross_entropy_loss(target) * 2, lambda a: numpy_cross_entropy(a, target) * 2, rnd(4, 3))

    print("--- zero_grad and step")
    A, B = rnd(2, 3), rnd(2, 3)

    def twice():
        a, b = Tensor(np.copy(A)), Tensor(np.copy(B))
        out = (a * b).sin()
        out.backward()
        out.zero_grad()
        zeroed = np.copy(a.grad)
        out.backward()
        return [zeroed, a.grad]

    def one_step():
        a, b = Tensor(np.copy(A), req_grad=True), Tensor(np.copy(B))
        out = a * b
        out.backward()
        out.step(learning_rate=0.1)
        return [a.data, b.data]

    check_values("backward, zero_grad, backward gives the same gradient as one backward", twice, [np.zeros_like(A), np.cos(A * B) * B])
    check_values("step changes only tensors with req_grad=True", one_step, [A - 0.1 * B, B])

    print(f"\n{failures} check(s) failed." if failures else "\nAll checks passed.")
    sys.exit(1 if failures else 0)
