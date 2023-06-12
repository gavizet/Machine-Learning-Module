import pytest
import numpy as np
from Module_05.ex_01.TinyStatistician import TinyStatistician

ARGS = [
    ([1, 42, 300, 10, 59]),
    ([1, 2, 3, 4, 5, 6, 7, 8, 9]),
    (np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])),
    ([-3423423, 4324234, 32423423, 7867, 7686, 90890, 5464]),
    ([199.99, 188.88, 177.77, 166.66, 155.55, 144.44, 133.33])
]


@pytest.mark.parametrize("array", ARGS)
def test_tiny_statistician(array):
    stats = TinyStatistician()
    np.testing.assert_equal(stats.mean(array), np.mean(array))
    np.testing.assert_equal(stats.median(array), np.median(array))
    np.testing.assert_equal(stats.quartile(array), [np.quantile(
        array, 0.25), np.quantile(array, 0.75)])
    # Using almost equal because I'm rounding down and numpy is rounding up at the 8th decimal
    np.testing.assert_almost_equal(
        stats.percentile(array, 15), np.percentile(array, 15))
    np.testing.assert_almost_equal(
        stats.percentile(array, 20), np.percentile(array, 20))
    np.testing.assert_almost_equal(
        stats.percentile(array, 80), np.percentile(array, 80))
    np.testing.assert_almost_equal(
        stats.percentile(array, 65), np.percentile(array, 65))

    np.testing.assert_equal(stats.var(array), np.var(array))
    np.testing.assert_equal(stats.std(array), np.std(array))
