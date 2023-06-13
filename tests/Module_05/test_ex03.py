import pytest
import numpy as np
from Module_05.ex_03.tools import add_intercept

PARAMS = [
    (np.arange(1, 6), np.array(
        [[1., 1.], [1., 2.], [1., 3.], [1., 4.], [1., 5.]])),
    (np.arange(1, 10).reshape((3, 3)), np.array(
        [[1., 1., 2., 3.], [1., 4., 5., 6.], [1., 7., 8., 9.]])),
]


@pytest.mark.parametrize("array, expected", PARAMS)
def test_add_intercept(array, expected):
    np.testing.assert_array_equal(add_intercept(array), expected)


ERRORS = [
    ("test", None),
    (12, None),
    ([1., 2., 3., 4., 5.], None),
    (np.array([]), None),
    (np.array([['1', '2'], ['3', '4']]), None),
]


@pytest.mark.parametrize("array, expected", ERRORS)
def test_simple_predict_errors(array, expected):
    result = add_intercept(array)
    assert result == None
