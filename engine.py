import numpy as np

# configure numpy to render floats with 3 decimal places
np.set_printoptions(formatter={"float": "{: 0.3f}".format})


# +++++++++++++++++ Assignment +++++++++++++++++
# In this file your task is to complete the functions
# marked with ellipses (...) and text cues. You should
# not change any other code in this file. You should
# also not import any other modules here.
#
# The goal of the task is to implement a custom tensor
# class that supports basic operations and automatic
# backpropagation.
# +++++++++++++++++++++++++++++++++++++++++++++++


def reshape_gradient(gradient: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Reduce a broadcast gradient back to the shape of the target Tensor.

    When numpy broadcasts a Tensor of shape `target_shape` to a larger shape,
    every element of the Tensor is used several times, so its gradient is the
    sum of the gradients of all its copies.

    Args:
        gradient: The gradient in the (broadcast) shape of the operation's output.
        target_shape: The shape of the target Tensor.

    Returns:
        The gradient summed down to `target_shape`.
    """
    gradient = np.asarray(gradient)

    # axes that broadcasting prepended to the target are summed away
    extra_dims = gradient.ndim - len(target_shape)
    gradient = np.sum(gradient, axis=tuple(range(extra_dims)))

    # axes where the target has size 1 but the gradient does not were stretched
    broadcast_axes = tuple(
        i
        for i, (grad_axis, tar_axis) in enumerate(zip(gradient.shape, target_shape))
        if tar_axis == 1 and grad_axis != 1
    )
    return np.sum(gradient, axis=broadcast_axes, keepdims=True)


def back_none():
    return None


class Tensor:
    """
    A custom tensor class that supports basic operations and automatic differentiation.

    Args:
        data (array-like): Input data to create the tensor.
        _parent (tuple, optional): Tuple of parent tensors in the computation graph. Defaults to ().
        _op (str, optional): Operation associated with this tensor. Defaults to ''.
        label (str, optional): Label or name for the tensor. Defaults to ''.
        req_grad (bool, optional): Whether gradient updates should be performed for this tensor. Defaults to False.
        is_weight (bool, optional): Whether the tensor has a batch dimension. Defaults to False. The batch dimension is the first dimension of the tensor.

    Attributes:
        data (numpy.ndarray): The underlying data stored in the tensor.
        label (str): A label for the tensor.
        grad (numpy.ndarray): Gradient of the tensor with respect to some loss.
        req_grad (bool): Indicates if gradient updates are to be performed for this tensor.
    """

    def __init__(
        self, data, _parent=(), _op="", label="", req_grad=False, is_weight=False
    ):
        self.data = np.array(data, dtype=np.float64)
        self.label = label
        self.grad = np.zeros(self.data.shape)
        self.req_grad = req_grad
        self.is_weight = is_weight
        self.grad_divisor = None

        self._backward = back_none
        self._prev = set(_parent)
        self._op = _op

    # +++++++++++++++++ Basic Operations +++++++++++++++++

    def __add__(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = ...  # 🌀 your code here

        def _backward():
            self.grad += reshape_gradient(..., self.data.shape)  # 🌀 your code here
            other.grad += reshape_gradient(..., other.data.shape)  # 🌀 your code here

        out._backward = _backward
        return out

    def __mul__(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = ...  # 🌀 your code here

        def _backward():
            self.grad += reshape_gradient(..., self.data.shape)  # 🌀 your code here
            other.grad += reshape_gradient(..., other.data.shape)  # 🌀 your code here

        out._backward = _backward
        return out

    def matmul(self, other) -> "Tensor":
        """Matrix product, like `np.matmul`. Both operands have at least 2
        dimensions; leading (batch) dimensions broadcast, e.g. (B, n, k) @ (k, m)."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = ...  # 🌀 your code here

        def _backward():
            self.grad += reshape_gradient(..., self.data.shape)  # 🌀 your code here
            other.grad += reshape_gradient(..., other.data.shape)  # 🌀 your code here

        out._backward = _backward
        return out

    def __pow__(self, other) -> "Tensor":
        assert isinstance(other, (int, float))
        out = ...  # 🌀 your code here

        def _backward():
            self.grad += ...  # 🌀 your code here

        out._backward = _backward

        return out

    def __sub__(self, other) -> "Tensor":
        return self + (-other)

    def __matmul__(self, other) -> "Tensor":
        return self.matmul(other)

    def __neg__(self) -> "Tensor":
        return self * -1

    def __truediv__(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * (other**-1)

    def __radd__(self, other) -> "Tensor":
        return self + other

    def __rsub__(self, other) -> "Tensor":
        return (-self) + other

    def __rmul__(self, other) -> "Tensor":
        return self * other

    def __rtruediv__(self, other) -> "Tensor":
        return other * (self**-1)

    def __rpow__(self, other) -> "Tensor":
        return (self * np.log(other)).exp()

    # +++++++++++++++++ Basic Functions +++++++++++++++++

    def sin(self) -> "Tensor":
        out = ...  # 🌀 your code here

        def _backward():
            self.grad += ...  # 🌀 your code here

        out._backward = _backward

        return out

    def cos(self) -> "Tensor":
        out = ...  # 🌀 your code here

        def _backward():
            self.grad -= ...  # 🌀 your code here

        out._backward = _backward

        return out

    def exp(self) -> "Tensor":
        out = ...  # 🌀 your code here

        def _backward():
            self.grad += ...  # 🌀 your code here

        out._backward = _backward

        return out

    def log(self) -> "Tensor":
        out = ...  # 🌀 your code here

        def _backward():
            self.grad += ...  # 🌀 your code here

        out._backward = _backward

        return out

    # +++++++++++++++++ Other Functions +++++++++++++++++

    def sum(self, axis=None) -> "Tensor":
        """Sum over `axis` (an int, or None for all elements), like `np.sum`."""
        out = ...  # 🌀 your code here

        def _backward():
            self.grad += ...  # 🌀 your code here

        out._backward = _backward

        return out

    def stack(self, other, axis=0) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            np.stack((self.data, other.data), axis=axis), (self, other), "stack"
        )

        def _backward():
            self.grad += np.take(out.grad, 0, axis=axis)
            other.grad += np.take(out.grad, 1, axis=axis)

        out._backward = _backward

        return out

    def T(self) -> "Tensor":
        out = Tensor(self.data.T, (self,), "T")

        def _backward():
            self.grad += out.grad.T

        out._backward = _backward

        return out

    # +++++++++++++++++ Activation Functions +++++++++++++++++

    def relu(self) -> "Tensor":
        out = ...  # 🌀 your code here

        def _backward():
            self.grad += ...  # 🌀 your code here

        out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        out = ...  # 🌀 your code here

        def _backward():
            self.grad += ...  # 🌀 your code here

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        out = ...  # 🌀 your code here

        def _backward():
            self.grad += ...  # 🌀 your code here

        out._backward = _backward
        return out

    # +++++++++++++++++ Loss Functions +++++++++++++++++

    def cross_entropy_loss(self, target: np.ndarray) -> "Tensor":
        """Mean cross-entropy of softmax(self) over the batch.

        `self` holds the logits with shape (N, C), `target` the N class indices.
        """
        assert (
            isinstance(target, np.ndarray) and len(target.shape) == 1
        ), "target must be a 1D numpy array"
        # TODO: write forward pass
        # Hints:
        # First compute the probabilities using the softmax function.
        # To ensure numerical stability it is recommended to substract
        # the highest number of every row from every element of that row.
        # Then construct the one hot vector encoding of targets.
        # Lastly compute the loss itself.
        # -------------------------------------------------
        # 🌀 INCEPTION 🌀 (Your code begins its journey here. 🚀 Do not delete this line.)
        #
        #                    ╔═══════════════════════╗
        #                    ║                       ║
        #                    ║       YOUR CODE       ║
        #                    ║                       ║
        #                    ╚═══════════════════════╝
        #

        # 🌀 TERMINATION 🌀 (Your code reaches its end. 🏁 Do not delete this line.)

        def _backward():
            self.grad += ...  # 🌀 TODO: your code here

        out._backward = _backward
        return out

    def regularization_loss(self, reg: float) -> "Tensor":
        """L2 regularization: `reg * sum(self ** 2)`."""
        out = ...  # 🌀 TODO: your code here

        def _backward():
            self.grad += ...  # 🌀 TODO: your code here

        out._backward = _backward
        return out

    # +++++++++++++++++ Backward Pass and Optimization +++++++++++++++++

    def backward(self) -> None:
        """Backpropagate from this Tensor through the whole graph below it.

        The gradient of this Tensor is set to ones (of its own shape), whatever
        it was before. Every other Tensor in the graph accumulates (+=) into its
        current gradient, so call `zero_grad` between two backward passes.
        """
        # TODO: write function to perform backward pass
        # -------------------------------------------------
        # 🌀 INCEPTION 🌀 (Your code begins its journey here. 🚀 Do not delete this line.)
        #
        #                    ╔═══════════════════════╗
        #                    ║                       ║
        #                    ║       YOUR CODE       ║
        #                    ║                       ║
        #                    ╚═══════════════════════╝
        #

        # 🌀 TERMINATION 🌀 (Your code reaches its end. 🏁 Do not delete this line.)

    def zero_grad(self) -> None:
        """Set the gradient of this Tensor and of every Tensor below it in the
        graph (intermediate results included) to zeros."""
        # TODO: write function to zero gradients
        # -------------------------------------------------
        # 🌀 INCEPTION 🌀 (Your code begins its journey here. 🚀 Do not delete this line.)
        #
        #                    ╔═══════════════════════╗
        #                    ║                       ║
        #                    ║       YOUR CODE       ║
        #                    ║                       ║
        #                    ╚═══════════════════════╝
        #

        # 🌀 TERMINATION 🌀 (Your code reaches its end. 🏁 Do not delete this line.)

    def step(self, learning_rate: float) -> None:
        """Gradient descent step: `data -= learning_rate * grad` for every Tensor
        in the graph that has `req_grad=True`. Other Tensors stay untouched."""
        # TODO: write function to perform a learning step
        # -------------------------------------------------
        # 🌀 INCEPTION 🌀 (Your code begins its journey here. 🚀 Do not delete this line.)
        #
        #                    ╔═══════════════════════╗
        #                    ║                       ║
        #                    ║       YOUR CODE       ║
        #                    ║                       ║
        #                    ╚═══════════════════════╝
        #

        # 🌀 TERMINATION 🌀 (Your code reaches its end. 🏁 Do not delete this line.)

    def _traverse_children(self) -> list:
        topo, visited = [], set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)

        build_topo(self)
        return topo

    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, grad={self.grad}, label={self.label})"
