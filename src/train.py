import numpy as np
import matplotlib.pyplot as plt

from load_csv import load
from predict import estimatePrice


def gradient_descent(X, Y, θ0, θ1, learningRate, iterations):
    m = len(X)  # Number of training examples

    for _ in range(iterations):
        sum_tmpθ0 = 0.0
        sum_tmpθ1 = 0.0

        for i in range(m):
            error = estimatePrice(X[i], θ0, θ1) - Y[i]
            sum_tmpθ0 += error
            sum_tmpθ1 += error * X[i]

        tmpθ0 = learningRate * (1/m) * sum_tmpθ0
        tmpθ1 = learningRate * (1/m) * sum_tmpθ1

        θ0 -= tmpθ0
        θ1 -= tmpθ1

    return θ0, θ1


def main():
    # collecting x & y
    df = load("../material/data.csv")
    X = df['km']
    Y = df['price']

    normalizedX = (X - X.mean()) / X.std()
    normalizedY = (Y - Y.mean()) / Y.std()

    # using the formula to calculate θ0 & θ1
    θ0 = θ1 = 1
    learningRate = 0.01
    iterations = 100000
    
    θ0_norm, θ1_norm = gradient_descent(normalizedX, normalizedY, θ0, θ1,
                                        learningRate, iterations)

    # Transforming θ0 and θ1 back to original scale
    θ1 = θ1_norm * (Y.std() / X.std())
    θ0 = θ0_norm * Y.std() + Y.mean() - θ1 * X.mean()

    print(f"theta0: {θ0}, theta1: {θ1}")

    # plotting values and regression line
    max_x = np.max(X) + 100
    min_x = np.min(X) - 100

    # calculating line values x and y
    x = np.linspace(min_x, max_x, 100)
    print(x)
    y = θ0 + θ1 * x
    print(f"theta0: {θ0}, theta1: {θ1}")

    plt.plot(x, y, color='#58b970', label='Regression Line')
    plt.scatter(X, Y, c='#ef5423', label='data points')

    plt.xlabel('Price')
    plt.ylabel('Mileage')
    plt.legend()
    plt.show()

    # calculate mean of x & y using an inbuilt numpy method mean()
    mean_x = np.mean(X)
    mean_y = np.mean(Y)

    # calculating R-squared value for measuring goodness of our model

    ss_t = 0  # total sum of squares
    ss_r = 0  # total sum of square of residuals

    for i in range(int(len(X))):
        y_pred = θ0 + θ1 * X[i]
        ss_t += (Y[i] - mean_y) ** 2
        ss_r += (Y[i] - y_pred) ** 2
    r2 = 1 - (ss_r/ss_t)

    print(r2)

    try:
        with open("../material/θ.csv", "w") as file:
            file.write(f'{ θ0 },{ θ1 }')
    except Exception as error:
        print(f'{type(error).__name__}: {error}')


if __name__ == "__main__":
    main()
