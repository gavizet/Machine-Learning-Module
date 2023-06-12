import numpy as np


class TinyStatistician:
    """ Class for basic statistic methods. All of them take a numpy.ndarray as input"""

    def __init__(self):
        pass

    def _is_valid_array(self, function):
        def wrapper(array: np.ndarray):
            """ Wrapper function for our _is_valid_array decorator """
            if not isinstance(array, np.ndarray) or len(array) == 0:
                return None
            result = function(self, array)
            return result
        return wrapper

    @_is_valid_array
    @staticmethod
    def mean(array) -> float or None:
        """Compute the mean of a given non-empty array.

        Args:
            array (numpy.ndarray): the given list we want to compute

        Returns:
            Mean of array as float or None if parameter is not valid / an error happened
        """

    @_is_valid_array
    @staticmethod
    def median(array) -> float or None:
        """Compute the median of a given non-empty array.

        Args:
            array (numpy.ndarray): the given list we want to compute

        Returns:
            Median of array as float or None if parameter is not valid / an error happened
        """
        pass

    @_is_valid_array
    @staticmethod
    def quartile(array) -> float or None:
        """Compute the 1st and 3rd quartile of a given non-empty array.

        Args:
            array (numpy.ndarray): the given list we want to compute

        Returns:
            Quartiles of array as list of 2 floats or None if parameter is not valid
        """
        pass

    @_is_valid_array
    @staticmethod
    def percentile(array, perc) -> float or None:
        """Compute the p percentage of a given non-empty array.

        Args:
            array (numpy.ndarray): the given list we want to compute
            perc (int): percentile we want to get

        Returns:
            Percentile of array as float or None if parameter is not valid / an error happened
        """
        pass

    @_is_valid_array
    @staticmethod
    def var(array) -> float or None:
        """Compute the variance of a given non-empty array.

        Args:
            array (numpy.ndarray): the given list we want to compute

        Returns:
            Variance of array as float or None if parameter is not valid / an error happened
        """
        pass

    @_is_valid_array
    @staticmethod
    def std(array) -> float or None:
        """Compute the standard deviation of a given non-empty array.

        Args:
            array (numpy.ndarray): the given list we want to compute

        Returns:
            Standard deviation of array as Float or 
            None if parameter is not valid / an error happened
        """
        pass
