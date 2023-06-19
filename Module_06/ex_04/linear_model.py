import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from Module_06.ex_03.my_linear_regression import MyLinearRegression as MyLR


class Fileloader:
    """ Loads a pandas dataframe from a csv file """

    def __init__(self):
        pass

    @staticmethod
    def load(path):
        """ Loads a CSV file from a path and returns the data as a pandas dataframe,

        Args:
          path (str): represents the file path of the CSV file that needs to be loaded.

        Returns:
          events (pd.DataFrame): contains the data from the CSV file
          None: if an error was found
        """
        events = None
        try:
            events = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            print("Empty csv file!")
        except (FileNotFoundError, ValueError):
            print("Csv file not found")
        return events


def main_tests():
    loader = Fileloader()
    data = loader.load("Module_06/ex_04/are_blue_pills_magics.csv")
    Xpill = np.array(data['Micrograms']).reshape(-1, 1)
    Yscore = np.array(data['Score']).reshape(-1, 1)

    linear_model1 = MyLR(np.array([[89.0], [-8]]))
    linear_model2 = MyLR(np.array([[89.0], [-6]]))

    Y_model1 = linear_model1.predict_(Xpill)
    Y_model2 = linear_model2.predict_(Xpill)

    print(linear_model1.mse_(Yscore, Y_model1))
    # 57.60304285714282
    print(mean_squared_error(Yscore, Y_model1))
    # 57.603042857142825
    print(linear_model1.mse_(Yscore, Y_model2))
    # 232.16344285714285
    print(mean_squared_error(Yscore, Y_model2))
    # 232.16344285714285


if __name__ == "__main__":
    main_tests()
