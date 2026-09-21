# HW2 - Autograd
In this assignment, it would be your task to implement your own autograd library, similar to the one shown in the lab. The difference is that while in the lab, we showed autograd for scalar values, your solution shall work with vectors and tensors as well.

## Task (10 pts)
The task is to complete the engine.py file. Here, you will find ellipses (…), where you should complete the functions to compute forward and backward passes correctly. There are also starting and ending with recognizable comments. Inside them, you should provide your own code. These are used for longer sections of code. In this homework, they are inside higher-level functions of the autograd.

You will be using the `numpy` library; its documentation may be found [here](https://numpy.org/doc/stable/index.html). Do NOT use any libraries with autograd, such as `PyTorch` or `Tensorflow`.

Most of the functions should be elementary. However, if you are unsure how to proceed with more complex ones, consult the lab materials before contacting tutors.

Please be aware that the variable `other` in function `__pow__` is either `int` or `float` and, therefore, cannot be put in the parent set of output. Backpropagation is available only for objects in the class `Tensor`.

Start with the `reshape_gradient` function at the top of the file. When numpy broadcasts an operand, e.g. a bias of shape `(3,)` added to a batch of shape `(2, 3)`, every element of the operand is used several times, and its gradient is the sum of the gradients of all its copies. `reshape_gradient` takes the gradient in the shape of the output and sums it down to the shape of the operand. Broadcasting does two things - it prepends axes to the operand of lower rank and it stretches axes of size 1 - and the function undoes them in two steps. Read the [broadcasting rules](https://numpy.org/doc/stable/user/basics.broadcasting.html) first; `test.py` checks this function on its own before anything else.

The function is only needed in the `__add__`, `__mul__` and `matmul` functions. Only in these three functions do we have the possibility of broadcasting (in `matmul` it is the leading batch dimensions that broadcast, e.g. `(B, n, k) @ (k, m)`), and all the other operations are built from them.

### What exactly is expected
The docstrings in `engine.py` are part of the assignment. The conventions that the evaluation relies on:

- The `grad` of every tensor always has the same shape as its `data`.
- `reshape_gradient(gradient, target_shape)` returns an array of exactly `target_shape`, e.g. `(2, 1)` and not `(2,)`. It is tested on its own as well.
- `backward()` sets the gradient of the tensor it is called on to ones and **accumulates** (`+=`) into all the other tensors of the graph. A tensor may be used several times in one expression (`a * a + a`).
- Every `_backward` has to take the gradient arriving from above (`out.grad`) into account, including `sum` and both loss functions - they are not always the last operation of the graph.
- `sum(axis)` works like `np.sum`, for `axis=None` as well as for an integer axis.
- `matmul` works like `np.matmul` for operands with at least two dimensions.
- `regularization_loss(reg)` is `reg * sum(x ** 2)`. `cross_entropy_loss(target)` takes logits of shape `(N, C)` and `N` class indices and returns the **mean** over the batch.
- `zero_grad()` zeroes the gradient of every tensor in the graph, intermediate results included. `step(learning_rate)` changes only tensors with `req_grad=True`.
- Numerical stability: `sigmoid` has to return finite, correct values even for inputs like ±800, and `cross_entropy_loss` for logits around ±1000 (see the hint in the code).
- `engine.py` may import only `numpy` (and `math`, `typing`). `eval`, `exec`, `open` and similar built-ins are rejected by the evaluation.

## Getting started
1. **Get files:** Either download the files as a zip or clone the repository:
```bash
git clone https://github.com/urob-ctu/hw2-autograd.git
```
2. **Install requirements:** There are some Python requirements you need to have installed:
```bash
pip install -r requirements.txt
```

3. **Complete homework:** You may work on the assignment.

## Testing
You may use the provided file `test.py` to test your implementation. It compares your tensors with plain numpy: values directly, gradients against numerical gradients (central differences). Functions that you have not implemented yet are reported as failed, so you can run it from the very beginning.

The grading uses the same kind of checks with different expressions, shapes and values, so the testing file is not exhaustive. Add your own cases at the bottom of `test.py` - one `check(...)` line each.

To run the tests, you may use the following command:
```bash
python test.py      # add -v for full tracebacks
```

Once everything passes, see your engine do what it was built for:
```bash
python demo.py
```
It trains a small neural network on three interleaved spirals using nothing but your `Tensor`. The loss has to fall and the accuracy should end above 95 %. With `matplotlib` installed it also saves the decision regions to `demo.png`. The grading contains a similar, much smaller training run.

## Submission and evaluation
Upload a zip file that contains your `engine.py` to BRUTE. You may use the provided bash script to create it:
```bash
./submit.sh
```

The maximum amount of points you may get is 10. The whole assignment will be auto-evaluated in BRUTE. The tutors may later re-evaluate any submission.

You will receive minus one point for every 24 hours after the deadline. However, no more than 9 points will be deducted for late submission.

Good luck, do not forget to play with the task a bit, and in the case of any questions or concerns please contact me.
