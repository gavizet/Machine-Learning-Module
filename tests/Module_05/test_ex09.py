import pytest
from math import sqrt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from Module_05.ex_09.other_losses import _args_are_valid, mse_, rmse_, mae_, r2score_

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
    # Valid args
    (np.array([0, 15, -9, 7, 12, 3, -21]),
     np.array([2, 14, -13, 5, 12, 4, -19]), True),
]


@pytest.mark.parametrize("x, y, expected", ERRORS)
def test_errors_other_loss_(x, y, expected):
    result = _args_are_valid(x, y)
    assert result == expected


ARGS = [
    # Test 1
    (np.array([[2.], [6.], [10.], [14.], [18.]]),
     np.array([[2.], [7.], [12.], [17.], [22.]])),
    # Test 2
    (np.array([2, 14, -13, 5, 12, 4, -19]).reshape(-1, 1),
     np.array([[0.], [15.], [-9.], [7.], [12.], [3.], [-21.]])),
    # Test 3
    (np.array([[0], [15], [-9], [7], [12], [3], [-21]]),
     np.array([[2], [14], [-13], [5], [12], [4], [-19]])),
    # Test 4
    (np.array([[34], [37], [44], [47], [48], [48], [46], [43], [32], [27], [26], [24]]),
     np.array([[37], [40], [46], [44], [46], [50], [45], [44], [34], [30], [22], [23]])),
    # Test 5
    (np.array([0, 15, -9, 7, 12, 3, -21]),
     np.array([2, 14, -13, 5, 12, 4, -19])),
]


@pytest.mark.parametrize("x, y", ARGS)
def test_no_loop_loss_(x, y):
    np.testing.assert_equal(mse_(x, y), mean_squared_error(x, y))
    np.testing.assert_equal(rmse_(x, y), sqrt(mean_squared_error(x, y)))
    np.testing.assert_equal(mae_(x, y), mean_absolute_error(x, y))
    np.testing.assert_equal(r2score_(x, y), r2_score(x, y))
