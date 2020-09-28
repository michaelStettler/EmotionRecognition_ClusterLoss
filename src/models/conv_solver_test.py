from __future__ import division
from sympy import *
from sympy import solve, Symbol, Matrix, solve_linear_system, solve_undetermined_coeffs, nsolve, solve_poly_system
from scipy import signal
import numpy as np

np.set_printoptions(precision=5, linewidth=400, suppress=True)

input = np.array([[0.3, .7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [.5, .1, .3, .2, .5, 0, 0, 0, 0, 0, 0, .2, 0],
             [0, 0, .2, .1, .7, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, .2, 0, 0],
             [0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0],
             [0, 0, .2, 0, 1, 1, 1, 1, 1, 1, 0, 0, .2],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, .2, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

# input = np.random.rand(10, 10)
print("input")
print(input)
print()

filter_2x2 = [[.5, .2],
              [.8, .4]]

y = signal.correlate2d(input, filter_2x2, mode='valid')
print("y")
print(y)
print()


def x1(i, j, w_00, w_10):
    return (input[i][j]*w_00 + input[i+1][j]*w_10)


w0_00, w0_10 = Symbol("w0_00"), Symbol("w0_10")

w1_00, w1_01 = Symbol("w1_00"), Symbol("w1_01")


f_00 = w1_00 * x1(0, 0, w0_00, w0_10)
f_01 = w1_01 * x1(0, 1, w0_00, w0_10)
print("f_00")
print(f_00)


# solve for w1
system = Matrix(((input[0,0], input[0,1], y[0,0]),
           (input[1,0], input[1,1], y[1,0])))

res = solve_linear_system(system, w1_00, w1_01)
# f_01 = w1_01 * x1(0, 1, w0_00, w0_10)
# f_02 = w1_00 * x1(0, 2, w0_00, w0_10)
# f_03 = w1_01 * x1(0, 3, w0_00, w0_10)
#
# res = solve([f_00, f_01, f_02, f_03],
#              w0_00, w0_10, w1_00, w1_01)
#
print(res)
w1_00 = res[w1_00]
w1_01 = res[w1_01]

# solve x1 (intermediate matrix)
x1_01, x1_11 = Symbol('x1_01'), Symbol('x1_11')

res_x1_01 = solve(w1_00 + x1_01 * w1_01 - y[0,0], x1_01)
res_x1_11 = solve(w1_00 + x1_11 * w1_01 - y[1,0], x1_11)

x1 = np.ones((2,2))
x1[0, 1] = res_x1_01[0]
x1[1, 1] = res_x1_11[0]

print(x1)

