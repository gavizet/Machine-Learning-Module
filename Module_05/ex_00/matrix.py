"""" Manipulation and understanding of basic matrix operations.
Numpy usage is forbidden"""


class Matrix:
    """" Matrix manipulation class"""

    @staticmethod
    def __check_args(arg) -> None:
        """Makes sure the args passed at initialization to our Matrix/Vector class is valid

        Args:
            arg (any): The arg we want to parse and check

        Raises:
            ValueError: if the arg is not either :
                - List of lists of equal length filled with int/float
                - Tuple of 2 positive int/float

        Returns:
            None : if not error was found while parsing
        """
        if not isinstance(arg, (list, tuple)):
            raise ValueError(
                f"Argument has to be a List or a Tuple, not {type(arg)}")
        if isinstance(arg, list):
            for index, elem in enumerate(arg):
                if not isinstance(elem, list):
                    raise ValueError(f"Argumnt has to be a List of List, \
                         not a List of {type(elem)}")
                if index == 0:
                    len_elem = len(elem)
                elif len_elem != len(elem) or len(elem) == 0:
                    raise ValueError(
                        "All sublists need to have the same length (> 0)")
                if not all(isinstance(num, (int, float)) for num in elem):
                    raise ValueError(
                        "All elements of the sublists need to be an int or a float")
        if isinstance(arg, tuple):
            if len(arg) != 2 or not \
                    all((isinstance(num, (int, float)) and num > 0) for num in arg):
                raise ValueError(
                    "Shape argument should be a Tuple of 2 positive int / float")

    @staticmethod
    def __get_data(arg) -> list:
        """Return the data for our Matrix/Vector instance at __init__

        Args:
            arg (list, tuple): what we base ourselves on to create the data

        Returns:
            list: If instance was initialized with a list, just return it.
                    If it was with a tuple, return a list of the dimensions specificied by
                    the tuple, and filled with '0.0' values
        """
        if isinstance(arg, list):
            data = arg
        else:
            # List of list of zeroes
            data = [[0.0] * arg[1]] * arg[0]
            return data
        return data

    @staticmethod
    def __get_shape(arg) -> tuple:
        """Return the shape for our Matrix/Vector instance at __init__

        Args:
            arg (list, tuple): what we base ourselves on to create the shape

        Returns:
            list: If instance was initialized with a tuple, just return it.
                If it was with a list, return a tuple that contains the dimensions of that list
        """
        if isinstance(arg, tuple):
            shape = arg
        else:
            # Tuple of 2 positive numbers
            shape = (len(arg), len(arg[0]))
        return shape

    def __init__(self, arg: list or tuple):
        self.__check_args(arg)
        self.data = self.__get_data(arg)
        self.shape = self.__get_shape(arg)
        self.type = type(self)

    def T(self):
        """Compute the Transpose of our Matrix

        Returns:
            Vector / Matrix: our new transposed object with the same type as self
        """
        result = [[self.data[col][row]
                   for col in range(self.shape[0])]
                  for row in range(self.shape[1])]
        return type(self)(result)

    def __add__(self, other):
        """Compute the addition of two Matrices of the same dimension

        Args:
            other (Matrix): the matrix we want to add to self

        Raises:
            ValueError: if other is not a matrix or is a matrix of different shape

        Returns:
            Matrix / Vector: our resulting object with the same type as self
        """
        if not isinstance(other, Matrix) or self.shape != other.shape:
            raise ValueError("Can only add between 2 Matrix of the same shape")
        result = [[self.data[row][col] + other.data[row][col]
                   for col in range(self.shape[1])]
                  for row in range(self.shape[0])]
        return type(self)(result)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        """Compute the substraction of two Matrices of the same dimension

        Args:
            other (Matrix): the matrix we want to sub to self

        Raises:
            ValueError: if other is not a matrix or is a matrix of different shape

        Returns:
            Matrix / Vector: our resulting object with the same type as self
        """
        if not isinstance(other, Matrix) or self.shape != other.shape:
            raise ValueError("Can only add between 2 Matrix of the same shape")
        result = [[self.data[row][col] - other.data[row][col]
                   for col in range(self.shape[1])]
                  for row in range(self.shape[0])]
        return type(self)(result)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        """Compute the multiplication of a Matrix by a Scalar/Vector/Matrix with valid dimensions

        Args:
            other (int / float / Vector / Matrix): the object we want to multiple our Matrix with

        Raises:
            ValueError: if the shapes of our objects are not compatible for multiplication
            TypeError: if the type of other is not an int / float / Vector / Matrix

        Returns:
            Matrix / Vector: our resulting object after multiplication
        """
        if isinstance(other, (int, float)):
            # Multiply each number of the Matrix by Scalar
            result = [[self.data[row][col] * other
                       for col in range(self.shape[1])]
                      for row in range(self.shape[0])]
            return type(self)(result)
        if isinstance(other, (Matrix, Vector)):
            # Can only multiply if n_col(self) == n_row(other)
            if self.shape[1] != other.shape[0]:
                raise ValueError(
                    "Wrong shape. Can only do AxB if the number of columns of A is equal to the number of rows of B.")
            # Result Shape is always (n_row(self), n_col(other)).
            # Matrix x Matrix => Matrix || Matrix x Vector => Vector
            # B(3, 1) * A(1, 3) => C(3, 3) || A(1, 3) * B(3, 1) => C(1, 1)
            if self.shape[0] == 1 or other.shape[1] == 1:
                result = Vector((self.shape[0], other.shape[1]))
            else:
                result = Matrix((self.shape[0], other.shape[1]))
            result = [[sum(self.data[self_row][neutral] * other.data[neutral][other_col]
                           for neutral in range(self.shape[1]))
                       for other_col in range(other.shape[1])]
                      for self_row in range(self.shape[0])]
            return type(self)(result)
        raise TypeError("Type should be a Matrix/Vector/Scalar")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        """Compute the division of a Matrix by a Scalar

        Args:
            other (int, float): the scalar by which we want to divide self

        Raises:
            ValueError: if other not an int / float, of it's == 0

        Returns:
            Matrix / Vector: our resulting object with the same type as self
        """
        if not isinstance(other, (int, float)):
            raise ValueError("Can only divide with int of float")
        if other == 0:
            raise ValueError("Cannot divide by 0 you pleb")
        result = [[self.data[row][col] / other
                   for col in range(self.shape[1])]
                  for row in range(self.shape[0])]
        return type(self)(result)

    def __rtruediv__(self, other):
        raise ArithmeticError(
            "Division of a scalar by a Vector/Matrix is not defined here.")

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"{type(self).__name__} - "\
            f"Shape: {self.shape}\n"\
            f"Values: {self.data}"


class Vector(Matrix):
    """ Vector manipulation class"""

    def __init__(self, arg):
        super().__init__(arg)
        if self.shape[0] != 1 and self.shape[1] != 1:
            raise ValueError("Not a valid vector shape.")

    def dot(self, other):
        """Compute the Dot Product between two vectors

        Args:
            other (Vector): the other Vector we want to do the dot product with

        Raises:
            ValueError: if other is not a Vector or its shape is different to self

        Returns:
            int / float: the result of the dot product
        """
        if not isinstance(other, Vector) or self.shape != other.shape:
            raise ValueError(
                "Can only do the dot product with 2 Vectors of the same shape")
        result = sum([self.data[row][col] * other.data[row][col]
                     for col in range(self.shape[1]) for row in range(self.shape[0])])
        return result

# Tested with pytest, just use this command from root folder of the project :
# 'pytest -vv tests/Module_05/test_ex00.py'
