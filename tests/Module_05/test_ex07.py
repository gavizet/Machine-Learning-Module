import pytest
import numpy as np
from Module_05.ex_07.vec_loss import loss_, _args_are_valid

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
def test_errors_no_loop_loss_(y, y_hat, expected):
    result = _args_are_valid(y, y_hat)
    assert result == expected


LOSS = [
    # Test 1
    (np.array([[2.], [6.], [10.], [14.], [18.]]),
     np.array([[2.], [7.], [12.], [17.], [22.]]),
     3.0),
    # Test 2
    (np.array([2, 14, -13, 5, 12, 4, -19]).reshape(-1, 1),
     np.array([[0.], [15.], [-9.], [7.], [12.], [3.], [-21.]]),
     2.142857142857143),
    # Test 3
    (np.array([[0], [15], [-9], [7], [12], [3], [-21]]),
     np.array([[2], [14], [-13], [5], [12], [4], [-19]]),
     2.142857142857143),
    # Test 4
    (np.array([[34], [37], [44], [47], [48], [48], [46], [43], [32], [27], [26], [24]]),
     np.array([[37], [40], [46], [44], [46], [50],
              [45], [44], [34], [30], [22], [23]]),
     2.9583333333333335),
]


@pytest.mark.parametrize("y, y_hat, expected", LOSS)
def test_no_loop_loss_(y, y_hat, expected):
    loss = loss_(y, y_hat)
    np.testing.assert_equal(loss, expected)
    # Perfect model, error should be 0.0
    np.testing.assert_equal(loss_(y, y), 0.0)
    np.testing.assert_equal(loss_(y_hat, y_hat), 0.0)
