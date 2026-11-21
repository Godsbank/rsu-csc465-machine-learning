"""Unit 1: derivatives, partial derivatives and the gradient.

We compare exact derivatives with numerical ones (the change in f
for a very small change in the input), then take a few steps
downhill on a function of two variables.
"""
import numpy as np


def f(w):                       # one variable: minimum at w = 3
    return (w - 3) ** 2


def g(v):                       # two variables: minimum at (1, -2)
    x, y = v
    return (x - 1) ** 2 + 2 * (y + 2) ** 2


def grad_g(v):                  # exact partial derivatives
    x, y = v
    return np.array([2 * (x - 1), 4 * (y + 2)])


h = 1e-6
print("f'(5) exact = 4, numerical =", round((f(5 + h) - f(5)) / h, 4))

# NORMAL CODE
# v = np.array([4.0, 1.0])
# num = [(g(v + h * e) - g(v)) / h for e in np.eye(2)]
# print("gradient of g at (4, 1): exact", grad_g(v),
#       "numerical", np.round(num, 3))

# for step in range(1, 6):        # gradient descent, learning rate 0.2
#     v = v - 0.6 * grad_g(v)
#     print(f"step {step}: v = {np.round(v, 3)}  g(v) = {g(v):.4f}")

# CODE FOR ASSIGNMENT IT WAS LOOPED
v0 = np.array([4.0, 1.0])

# This loop tries one learning rate at a time: 0.05, 0.4, and 0.6.
# The learning rate controls how large each gradient-descent step is.
# Start from (4, 1) again for each rate so earlier results do not affect
# the next rate.
for learning_rate in (0.05, 0.4, 0.6):
    v = v0.copy()

    print(f"\nLearning rate: {learning_rate}")

    for step in range(1, 6):
        # The gradient points toward higher values of g.
        # Subtracting it moves the point toward the minimum at (1, -2).
        v = v - learning_rate * grad_g(v)
        print(f"step {step}: v = {np.round(v, 3)}  g(v) = {g(v):.4f}")

# 0.05: converges slowly because it takes small steps toward (1, -2).
# 0.4: converges fastest among the three rates and gets close to (1, -2).
# 0.6: diverges because the steps grow larger instead of approaching (1, -2).

# The second derivatives of g are 2 in the x direction and 4 in the y
# direction. In gradient descent, the error adjustment factors are:
#   x: 1 - 2 * learning_rate
#   y: 1 - 4 * learning_rate
#
# At 0.05: the factors are 0.90 and 0.80. Both are less than 1 in magnitude,
# so the distance from the minimum gets smaller slowly.
#
# At 0.4: the factors are 0.20 and -0.60. Both are less than 1 in magnitude,
# so the distance from the minimum gets smaller. The negative y factor
# makes the y value oscillate around -2.
#
# At 0.6: the factors are -0.20 and -1.40. The y factor has a magnitude
# greater than 1, so it makes the error grow and causes divergence.