import pytest
import numpy as np
from Module_06.ex_00.gradient import simple_gradient, _args_are_valid

ERRORS = [
    # Type error
    ([[2, 4, 6, 8, 10]], [[2, 4, 6, 8, 10]], [2, 0.7], False),
    # Elem type error
    (np.array([['1'], ['2']]), np.array(
        [[2], [4]]), np.array([2, 0.7]), False),
    # Arrays not same size error
    (np.array([[2]]), np.array([[2], [4]]), np.array([2, 0.7]), False),
    # Empty arrays error
    (np.array([[]]), np.array([[]]), np.array([2, 0.7]), False),
    # Shape error
    (np.array([[2, 4, 6, 8, 10]]), np.array(
        [[2, 4, 6, 8, 10]]), np.array([2, 0.7]), False),
    # Theta wrong shape
    (np.array([[2.], [6.], [10.], [14.], [18.]]),
     np.array([[2.], [7.], [12.], [17.], [22.]]), np.array([2]), True),
    # Valid args
    (np.array([[2.], [6.], [10.], [14.], [18.]]),
     np.array([[2.], [7.], [12.], [17.], [22.]]), np.array([2, 0.7]), True),
    # Valid args
    (np.array([0, 15, -9, 7, 12, 3, -21]),
     np.array([2, 14, -13, 5, 12, 4, -19]), np.array([2, 0.7]), True),
]


@pytest.mark.parametrize("x, y, theta, expected", ERRORS)
def test_errors_gradient(x, y, theta, expected):
    result = _args_are_valid(x, y, theta)
    assert result == expected


x_1 = np.array([12.4956442, 21.5007972, 31.5527382,
                48.9145838, 57.5088733]).reshape((-1, 1))
y_1 = np.array([37.4013816, 36.1473236, 45.7655287,
                46.6793434, 59.5585554]).reshape((-1, 1))

theta1 = np.array([2, 0.7]).reshape((-1, 1))
theta2 = np.array([1, -0.4]).reshape((-1, 1))
theta3 = np.array([3, 2]).reshape((-1, 1))
theta4 = np.array([2.4, 1.1]).reshape((-1, 1))

GRADIENTS = [
    (x_1, y_1, theta1, np.array([[-19.0342574], [-586.66875564]])),
    (x_1, y_1, theta2, np.array([[-57.86823748], [-2230.12297889]])),
    (x_1, y_1, theta3, np.array([[26.67862814], [1349.34177596]])),
    (x_1, y_1, theta4, np.array([[-4.87644647], [12.201672]])),
]


@pytest.mark.parametrize("x, y, theta, expected", GRADIENTS)
def test_gradient(x, y, theta, expected):
    np.testing.assert_array_almost_equal(simple_gradient(x, y, theta),
                                         expected)
