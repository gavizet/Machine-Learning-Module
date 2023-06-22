import pytest
import numpy as np
from Module_07.ex_03.gradient import gradient

# x, matrix (7, 3)
x = np.array([[-6, -7, -9],
              [13, -2, 14],
              [-7, 14, -1],
              [-8, -4, 6],
              [-5, -9, 6],
              [1, -5, 11],
              [9, -11, 8]])
# y, vector (6, 1)
bad_y = np.array([2, 14, -13, 5, 12, 4]).reshape((-1, 1))
# theta, vector (4, 1)
bad_theta = np.array([3, 0.5, -6, 4]).reshape((-1, 1))

ERRORS = [
    # Type error
    ([[2, 4, 6, 8, 10]], [[2, 4, 6, 8, 10]], [2, 0.7], None),
    # Elem type error
    (np.array([['1'], ['2']]), np.array(
        [[2], [4]]), np.array([2, 0.7]), None),
    # Arrays not same size error
    (np.array([[2]]), np.array([[2], [4]]), np.array([2, 0.7]), None),
    # Empty arrays error
    (np.array([[]]), np.array([[]]), np.array([2, 0.7]), None),
    # Shape error
    (np.array([[2, 4, 6, 8, 10]]), np.array(
        [[2, 4, 6, 8, 10]]), np.array([2, 0.7]), None),
    # y nrows != x nrows
    (x, bad_y, np.array([3, 0.5, -6]).reshape((-1, 1)), None),
    # theta nrows != x ncols
    (x, np.array([2, 14, -13, 5, 12, 4]).reshape((-1, 1)), bad_theta, None),
]


@pytest.mark.parametrize("x, y, theta, expected", ERRORS)
def test_errors_vec_gradient(x, y, theta, expected):
    result = gradient(x, y, theta)
    assert result == expected


y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
theta1 = np.array([3, 0.5, -6]).reshape((-1, 1))
theta2 = np.array([0, 0, 0]).reshape((-1, 1))

GRADIENTS = [
    # Example 1
    (x, y, theta1, np.array([[-33.71428571],
                             [-37.35714286],
                             [183.14285714],
                             [-393.]])),
    # Example 2
    (x, y, theta2, np.array([[-0.71428571],
                             [0.85714286],
                             [23.28571429],
                             [-26.42857143]])),
]


@pytest.mark.parametrize("x, y, theta, expected", GRADIENTS)
def test_gradient_vectorized(x, y, theta, expected):
    np.testing.assert_array_almost_equal(gradient(x, y, theta),
                                         expected)
