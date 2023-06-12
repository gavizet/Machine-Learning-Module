""" Module for basic stat computations """
import numpy as np


class TinyStatistician:
    """ Class for basic statistic methods. All of them take a numpy.ndarray as input"""

    def __init__(self):
        pass

    def _is_valid_array(function):
        def wrapper(self, array, *args, **kwargs):
            """ Wrapper function for our _is_valid_array decorator 
            Make sure the array passed to our method is valid """
            if not isinstance(array, (np.ndarray, list)) or len(array) == 0:
                return None
            # Make nparray a python list so we can parse it the same way
            if isinstance(array, np.ndarray):
                array = array.tolist()
            if not all(isinstance(num, (int, float)) for num in array):
                return None
            args_num = len(args)
            if args_num == 2:
                perc = args[1]
                # Make sure perc is a valid argument
                if not isinstance(perc, int) or not 0 <= perc <= 100:
                    return None
            result = function(self, array, *args, **kwargs)
            return result
        return wrapper

    @_is_valid_array
    def mean(self, array) -> float or None:
        """Compute the mean of a given non-empty array.

        Args:
            array (numpy.ndarray): the given list we want to compute

        Returns:
            Mean of array as float or None if parameter is not valid / an error happened
        """
        array_len = len(array)
        result = sum([elem for elem in array]) / array_len
        return float(result)

    @_is_valid_array
    def median(self, array) -> float or None:
        """Compute the median of a given non-empty array.

        Args:
            array (numpy.ndarray): the given list we want to compute

        Returns:
            Median of array as float or None if parameter is not valid / an error happened
        """
        return self.percentile(array, 50)

    @_is_valid_array
    def quartile(self, array) -> float or None:
        """Compute the 1st and 3rd quartile of a given non-empty array.

        Args:
            array (numpy.ndarray): the given list we want to compute

        Returns:
            Quartiles of array as list of 2 floats or None if parameter is not valid
        """
        result = [self.percentile(
            array, 25), self.percentile(array, 75)]
        return result

    @_is_valid_array
    def percentile(self, array, perc) -> float or None:
        """Compute the p percentile of a given non-empty array.
            We assume we want linear interpolation between the two closest list element

        Args:
            array (numpy.ndarray): the given list we want to compute
            perc (int): percentile we want to get

        Returns:
            Percentile of array as float or None if parameter is not valid / an error happened
        """
        array.sort()
        len_array = len(array) - 1
        index = (len_array) * (perc / 100)
        if isinstance(index, int):
            return array[index]
        floor = int(index)
        ceiling = floor + 1
        result = array[floor] + \
            ((index - floor) * (array[ceiling] - array[floor]))
        return result

    @_is_valid_array
    def var(self, array) -> float or None:
        """Compute the variance of a given non-empty array.

        Args:
            array (numpy.ndarray): the given list we want to compute

        Returns:
            Variance of array as float or None if parameter is not valid / an error happened
        """
        result = 0
        elem_num = len(array)
        mean = self.mean(array)
        result = sum([((num - mean) ** 2) for num in array])
        return result / elem_num

    @_is_valid_array
    def std(self, array) -> float or None:
        """Compute the standard deviation of a given non-empty array.

        Args:
            array (numpy.ndarray): the given list we want to compute

        Returns:
            Standard deviation of array as Float or 
            None if parameter is not valid / an error happened
        """
        return self.var(array) ** 0.5


if __name__ == "__main__":
    a = np.array([1, 42, 300, 10, 59])

    stats = TinyStatistician()
    assert stats.mean(a) == np.mean(a)
    assert stats.median(a) == np.median(a)
    assert stats.quartile(
        a) == [np.quantile(a, 0.25), np.quantile(a, 0.75)]
    np.testing.assert_almost_equal(
        stats.percentile(a, 15), np.percentile(a, 15))
    np.testing.assert_almost_equal(
        stats.percentile(a, 20), np.percentile(a, 20))
    np.testing.assert_almost_equal(
        stats.percentile(a, 80), np.percentile(a, 80))
    np.testing.assert_almost_equal(stats.var(a), np.var(a))
    np.testing.assert_almost_equal(stats.std(a), np.std(a))
    # See tests/Module_05/test_ex01.py for full battery of tests with pytest
