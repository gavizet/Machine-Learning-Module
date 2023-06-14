import numpy as np
import matplotlib.pyplot as plt
from Module_05.ex_07.vec_loss import loss_
from Module_05.ex_04.prediction import predict_


def plot_with_loss(x, y, theta):
    """Plot the data, prediction line and loss from three non-empty numpy.ndarray.

    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * 1.
        y: has to be an numpy.ndarray, a vector of dimension m * 1.
        theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.

    Returns:
        Nothing.

    Raises:
        This function should not raise any Exception.
    """
    # No need for any error handling since predict_ and loss_ function already do it
    prediction = predict_(x, theta)
    loss = loss_(y, prediction) * 2
    plt.figure()
    plt.title(f'Cost: {loss:.6f}')
    plt.scatter(x, y, marker='o')
    plt.plot(x, prediction, color='orange')
    for i, example in enumerate(x):
        # Plots a line starting at coordinate (x[i], y[i]) and going to (x[i], prediction[i])
        # for every example (input value)
        plt.plot((example, example),
                 (y[i], prediction[i]), color='red', linestyle='--')
    plt.show()


def main():
    x = np.arange(1, 6)
    y = np.array([11.52434424, 10.62589482,
                 13.14755699, 18.60682298, 14.14329568])
    # Example 1:
    theta1 = np.array([18, -1])
    plot_with_loss(x, y, theta1)

    # Example 2:
    theta2 = np.array([14, 0])
    plot_with_loss(x, y, theta2)

    # Example 3:
    theta3 = np.array([12, 0.8])
    plot_with_loss(x, y, theta3)


if __name__ == "__main__":
    main()
