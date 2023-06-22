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
