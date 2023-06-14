import pytest
import numpy as np
from Module_05.ex_06.loss import loss_, loss_elem_, _args_are_valid

ERRORS = [
    # Type error
    ([[2, 4, 6, 8, 10]], [[2, 4, 6, 8, 10]], False),
    # Elem type error
    (np.array([['1'], ['2']]), np.array([[2], [4]]), False),
    # Arrays not same size error
    (np.array([[2]]), np.array([[2], [4]]), False),
    # Empty arrays error
    (np.array([[]]), np.array([[]]), False),
    # Shape error
    (np.array([[2, 4, 6, 8, 10]]), np.array([[2, 4, 6, 8, 10]]), False),
    # Valid args
    (np.array([[2.], [6.], [10.], [14.], [18.]]),
     np.array([[2.], [7.], [12.], [17.], [22.]]), True),
]


@pytest.mark.parametrize("y, y_hat, expected", ERRORS)
def test_errors_predict_(y, y_hat, expected):
    result = _args_are_valid(y, y_hat)
    assert result == expected


LOSS_ELEM = [
    # Test 1
    (np.array([[2.], [6.], [10.], [14.], [18.]]),
     np.array([[2.], [7.], [12.], [17.], [22.]]),
     np.array([[0.], [1], [4], [9], [16]])),
    # Test 2
    (np.array([2, 14, -13, 5, 12, 4, -19]).reshape(-1, 1),
     np.array([[0.], [15.], [-9.], [7.], [12.], [3.], [-21.]]),
     np.array([[4.], [1.], [16.], [4.], [0.], [1.], [4.]])),
]


@pytest.mark.parametrize("y, y_hat, expected", LOSS_ELEM)
def test_loss_elem_(y, y_hat, expected):
    loss_elem = loss_elem_(y, y_hat)
    np.testing.assert_array_equal(loss_elem, expected)


LOSS = [
    # Test 1
    (np.array([[2.], [6.], [10.], [14.], [18.]]),
     np.array([[2.], [7.], [12.], [17.], [22.]]),
     3.0),
    # Test 2
    (np.array([2, 14, -13, 5, 12, 4, -19]).reshape(-1, 1),
     np.array([[0.], [15.], [-9.], [7.], [12.], [3.], [-21.]]),
     2.142857142857143),
]


@pytest.mark.parametrize("y, y_hat, expected", LOSS)
def test_loss_(y, y_hat, expected):
    loss = loss_(y, y_hat)
    np.testing.assert_equal(loss, expected)
    np.testing.assert_equal(loss_(y, y), 0.0)
    np.testing.assert_equal(loss_(y_hat, y_hat), 0.0)
