import pytest
import numpy as np
from Module_07.ex_02.loss import loss_

TRAINING_EX1 = np.arange(1, 13).reshape((4, -1))

# y, y_hat, result
ERRORS = [
    # Type error
    ([[2, 4, 6, 8, 10]], [[2, 4, 6, 8, 10]], None),
    # Elem type error
    (TRAINING_EX1, np.array([['5.'], ['5.'], ['5.'], ['5.']]), None),
    # Arrays not same size error
    (TRAINING_EX1, np.array([[2], [4]]), None),
    # Empty array error
    (TRAINING_EX1, np.array([[], [], []]), None),
    # Theta Shape error
    (TRAINING_EX1, np.array([[2, 4, 6, 8, 10]]), None),
]


@pytest.mark.parametrize("y, y_hat, expected", ERRORS)
def test_errors_vect_loss_(y, y_hat, expected):
    result = loss_(y, y_hat)
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
def test_vec_loss_(y, y_hat, expected):
    loss = loss_(y, y_hat)
    np.testing.assert_equal(loss, expected)
    # Perfect model, error should be 0.0
    np.testing.assert_equal(loss_(y, y), 0.0)
    np.testing.assert_equal(loss_(y_hat, y_hat), 0.0)
