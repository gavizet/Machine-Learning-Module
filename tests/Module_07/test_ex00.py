import pytest
import numpy as np
from Module_07.ex_00.prediction import simple_predict

TRAINING_EX1 = np.arange(1, 13).reshape((4, -1))

# X, theta, result
ERRORS = [
    # Type error
    ([[2, 4, 6, 8, 10]], [[2, 4, 6, 8, 10]], False),
    # Elem type error
    (TRAINING_EX1, np.array([['5.'], ['5.'], ['5.'], ['5.']]), False),
    # ncols training_ex1 != nrows theta error
    (TRAINING_EX1, np.array([[5.], [5.], [5.]]), False),
    # Empty array error
    (TRAINING_EX1, np.array([[], [], []]), False),
    # Shape error
    (TRAINING_EX1, False),
    # Theta wrong shape
    (TRAINING_EX1, False),
    # Valid args
    (TRAINING_EX1, True),
    # Valid args
    (TRAINING_EX1, True),
]
