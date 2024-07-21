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
    # collecting X and Y
    try:
        df = load("../material/data.csv")
    except Exception:
        print('The database could not be retrieved.')
        return 1

    X = df['km']
    Y = df['price']

    normalizedX = (X - X.mean()) / X.std()
    normalizedY = (Y - Y.mean()) / Y.std()

    # using the formula to calculate θ0 & θ1
    θ0 = θ1 = 1
    learningRate = 0.01
    iterations = 10000

    θ0_norm, θ1_norm = gradient_descent(normalizedX, normalizedY, θ0, θ1,
                                        learningRate, iterations)

    # Transforming θ0 and θ1 back to original scale
    θ1 = θ1_norm * (Y.std() / X.std())
    θ0 = θ0_norm * Y.std() + Y.mean() - θ1 * X.mean()

    # plotting values and regression line
    max_x = np.max(X) + 100
    min_x = np.min(X) - 100

    # calculating line values x and y
    x = np.linspace(min_x, max_x, 100)
    y = θ0 + θ1 * x

    plt.plot(x, y, color='#58b970', label='Regression Line')
    plt.scatter(X, Y, c='#ef5423', label='data points')

    plt.xlabel('Price')
    plt.ylabel('Mileage')
    plt.legend()
    plt.show()

    try:
        with open("../material/θ.csv", "w") as file:
            file.write(f'{ θ0 },{ θ1 }')
    except Exception as error:
        print(f'{type(error).__name__}: {error}')


if __name__ == "__main__":
    main()
