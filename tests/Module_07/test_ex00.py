import pytest
import numpy as np
from Module_07.ex_00.prediction import simple_predict

TRAINING_EX1 = np.arange(1, 13).reshape((4, -1))

# X, theta, result
ERRORS = [
    # Type error
    ([[2, 4, 6, 8, 10]], [[2, 4, 6, 8, 10]], None),
    # Elem type error
    (TRAINING_EX1, np.array([['5.'], ['5.'], ['5.'], ['5.']]), None),
    # ncols training_ex1 != nrows theta error
    (TRAINING_EX1, np.array([[5.], [5.], [5.]]), None),
    # Empty array error
    (TRAINING_EX1, np.array([[], [], []]), None),
    # Theta Shape error
    (TRAINING_EX1, np.array([[2, 4, 6, 8, 10]]), None),
]


@pytest.mark.parametrize("x, theta, expected", ERRORS)
def test_errors_iterative_simple_predict(x, theta, expected):
    result = simple_predict(x, theta)
    assert result == expected


THETA1 = np.array([5, 0, 0, 0]).reshape((-1, 1))
THETA2 = np.array([0, 1, 0, 0]).reshape((-1, 1))
THETA3 = np.array([-1.5, 0.6, 2.3, 1.98]).reshape((-1, 1))
THETA4 = np.array([-3, 1, 2, 3.5]).reshape((-1, 1))

ARGS = [
    # Test 1
    (TRAINING_EX1, THETA1, np.array([[5.], [5.], [5.], [5.]])),
    # Test 2
    (TRAINING_EX1, THETA2, np.array([[1.], [4.], [7.], [10.]])),
    # Test 3
    (TRAINING_EX1, THETA3, np.array([[9.64], [24.28], [38.92], [53.56]])),
    # Test 4
    (TRAINING_EX1, THETA4, np.array([[12.5], [32.], [51.5], [71.]])),
]


@pytest.mark.parametrize("x, theta, expected", ARGS)
def test_iterative_simple_predict(x, theta, expected):
    np.testing.assert_almost_equal(simple_predict(x, theta), expected)
