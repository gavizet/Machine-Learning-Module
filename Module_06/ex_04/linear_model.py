""" Evaluate a linear regression model on a very small dataset, with a given hypothesis h. 
Manipulate the loss function J, plot it, and briefly analyze the plot. """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from Module_06.ex_03.my_linear_regression import MyLinearRegression as MyLR


def load_csv(path: str) -> pd.DataFrame | None:
    """ Loads a CSV file from a path and returns the data as a pandas dataframe,

    Args:
      path (str): represents the file path of the CSV file that needs to be loaded.

    Returns:
      events (pd.DataFrame): contains the data from the CSV file
      None: if the file was not found or is empty
    """
    events = None
    try:
        events = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        print("Empty csv file!")
    except (FileNotFoundError, ValueError):
        print("Csv file not found")
    return events


def plot_regression(x: np.ndarray, y: np.ndarray, y_hat: np.ndarray):
    """ Plot the data and the hypothesis we get for 
        y (space driving score) versus x (quantity of blue pills)

    Args:
        x (np.ndarray): vector representing our quantity of blue pills
        y (np.ndarray): vector representing our space driving score
        y_hat (np.ndarray): vector representing our predicted values
    """
    # Plot our predicted values
    plt.plot(x, y_hat, "--X", color="limegreen", label='$S_{predict}(pills)$')
    # Plot our real values
    plt.scatter(x, y, marker="o", color="cyan", label='$S_{true}$(pills)')
    plt.ylabel("Space driving score")
    plt.xlabel("Quantity of blue pills (in micrograms)")
    plt.grid(visible=True)
    # See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html
    # Position the legend at the top of the plot (lower left, x_axis=0, y_axis=1)
    plt.legend(ncols=2, loc='lower left', bbox_to_anchor=(0, 1), frameon=False)
    plt.show()


def plot_loss(x: np.ndarray, y: np.ndarray):
    """ Plot the evolution of the loss function J(θ) as a function of θ1 
        for different values of θ0.

    Args:
        x (np.ndarray): vector representing our quantity of blue pills
        y (np.ndarray): vector representing our space driving score
    """
    size = 80
    plt.ylim(10, 150)
    plt.xlim(-14.4, -3.4)
    theta0 = np.linspace(81, 97, 6)
    theta1 = np.linspace(-14, -4, size)
    # Loop between all theta0 values
    for t0_index, t0_value in enumerate(theta0):
        # Create a vector full of zeros with the same number of values as theta1 where
        # we will store all the loss results for all values of theta1
        all_loss = np.zeros((size,))
        # Loop between all theta1 values
        for index, t1_value in enumerate(theta1):
            # Create instance of MyLinearRegression with our current theta values
            my_lin_reg = MyLR(np.array([[t0_value], [t1_value]]))
            # Get predictions
            y_hat = my_lin_reg.predict_(x)
            # For the current value of thetas, compute the loss and replace that value in the
            # all loss array
            all_loss[index] = my_lin_reg.loss_(y, y_hat)
        # Plot the error curve for the current value of theta0 and all values of theta1
        plt.plot(theta1, all_loss, linewidth=2, label=f"J((θ0=c{t0_index},θ1)")
    # Add info to the plot
    plt.xlabel('$θ_1$')
    plt.ylabel('Cost function $J(θ_0,θ_1)$')
    plt.grid(visible=True)
    plt.legend(loc="lower right")
    plt.show()


def main():
    # Get data from CSV file
    data = load_csv("Module_06/ex_04/are_blue_pills_magics.csv")
    # Store Micrograms values in vector x
    x = np.array(data['Micrograms']).reshape(-1, 1)
    # Store Score values in vector y
    y = np.array(data['Score']).reshape(-1, 1)

    my_lin_reg = MyLR(np.array([[89.0], [-8]]))

    # Get prediction, loss and thetas before training
    y_hat1 = my_lin_reg.predict_(x)
    print(f"MSE before fit: {my_lin_reg.mse_(y, y_hat1)}")
    print(f'Thetas before training: {my_lin_reg.thetas}')

    # Train the model and et prediction, loss and thetas
    my_lin_reg.fit_(x, y)
    y_hat1 = my_lin_reg.predict_(x)
    print(f"MSE after fit: {my_lin_reg.mse_(y, y_hat1)}")
    print(f'Thetas after training: {my_lin_reg.thetas}')

    plot_regression(x, y, y_hat1)
    plot_loss(x, y)


if __name__ == "__main__":
    main()
