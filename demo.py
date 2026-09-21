#!/usr/bin/env python
"""Trains a small neural network with your engine: python demo.py

Three interleaved spirals, a network with one hidden layer, plain gradient
descent. Nothing here is graded - it is the reason why the engine exists. Once
engine.py is complete, the loss falls and the accuracy ends above 95 %.
If matplotlib is installed, the decision regions are saved to demo.png.
"""

import numpy as np

from engine import Tensor

CLASSES, POINTS, HIDDEN = 3, 100, 32
EPOCHS, LEARNING_RATE, REG = 1000, 1.0, 1e-4


def spirals(rng):
    radius = np.linspace(0.05, 1, POINTS)
    x, y = [], []
    for c in range(CLASSES):
        angle = np.linspace(0, 4, POINTS) + c * 2 * np.pi / CLASSES + rng.normal(scale=0.15, size=POINTS)
        x.append(np.stack((radius * np.sin(angle), radius * np.cos(angle)), axis=1))
        y.append(np.full(POINTS, c))
    return np.concatenate(x), np.concatenate(y)


def logits(x, w1, b1, w2, b2):
    return (x @ w1 + b1).relu() @ w2 + b2  # b1, b2 are broadcast over the batch


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x, y = spirals(rng)
    w1 = Tensor(rng.normal(size=(2, HIDDEN)) * np.sqrt(2 / 2), req_grad=True)
    b1 = Tensor(np.zeros(HIDDEN), req_grad=True)
    w2 = Tensor(rng.normal(size=(HIDDEN, CLASSES)) * np.sqrt(2 / HIDDEN), req_grad=True)
    b2 = Tensor(np.zeros(CLASSES), req_grad=True)

    for epoch in range(EPOCHS + 1):
        out = logits(Tensor(x), w1, b1, w2, b2)
        loss = out.cross_entropy_loss(y) + w1.regularization_loss(REG) + w2.regularization_loss(REG)
        if epoch % 100 == 0:
            accuracy = np.mean(np.argmax(out.data, axis=1) == y)
            print(f"epoch {epoch:4d}   loss {float(loss.data):.4f}   accuracy {accuracy:.1%}")
        loss.zero_grad()
        loss.backward()
        loss.step(learning_rate=LEARNING_RATE)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        pass
    else:
        grid = np.stack(np.meshgrid(np.linspace(-1.1, 1.1, 300), np.linspace(-1.1, 1.1, 300)), axis=-1)
        regions = np.argmax(logits(Tensor(grid.reshape(-1, 2)), w1, b1, w2, b2).data, axis=1).reshape(300, 300)
        plt.contourf(grid[..., 0], grid[..., 1], regions, levels=np.arange(CLASSES + 1) - 0.5, alpha=0.3, cmap="brg")
        plt.scatter(x[:, 0], x[:, 1], c=y, s=12, cmap="brg")
        plt.gca().set_aspect("equal")
        plt.savefig("demo.png", dpi=120)
        print("decision regions saved to demo.png")
