import numpy as np
import matplotlib.pyplot as plt

from load_csv import load
from predict import estimatePrice


def linePlotter(X, Y, θ0_norm, θ1_norm, label=False):
    # Transforming θ0 and θ1 back to original scale
    θ0, θ1 = unnormalizeΘ(X, Y, θ0_norm, θ1_norm)

    # plotting values and regression line
    max_x = np.max(X) + 100
    min_x = np.min(X) - 100

    # calculating line values x and y
    x = np.linspace(min_x, max_x, 100)
    y = θ0 + θ1 * x

    if label:
        plt.plot(x, y, label='Regression Line')
    else:
        plt.plot(x, y)


def gradientDescent(X, Y, θ0, θ1, learningRate, iterations, printInterval):
    m = len(X)  # Number of population values

    normalizedX = (X - X.mean()) / X.std()
    normalizedY = (Y - Y.mean()) / Y.std()

    for i in range(iterations):
        sum_tmpθ0 = 0.0
        sum_tmpθ1 = 0.0

        for j in range(m):
            error = estimatePrice(normalizedX[j], θ0, θ1) - normalizedY[j]
            sum_tmpθ0 += error
            sum_tmpθ1 += error * normalizedX[j]

        tmpθ0 = learningRate * (1/m) * sum_tmpθ0
        tmpθ1 = learningRate * (1/m) * sum_tmpθ1

        θ0 -= tmpθ0
        θ1 -= tmpθ1

        if i % printInterval == 0:
            yield θ0, θ1


def unnormalizeΘ(X, Y, θ0_norm, θ1_norm):
    θ1 = θ1_norm * (Y.std() / X.std())
    θ0 = θ0_norm * Y.std() + Y.mean() - θ1 * X.mean()
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

    # using the formula to calculate θ0 & θ1
    θ0 = θ1 = 1
    learningRate = 0.01
    iterations = 10000

    for θ0_norm, θ1_norm in gradientDescent(X, Y, θ0, θ1, learningRate,
                                            iterations, 50):
        linePlotter(X, Y, θ0_norm, θ1_norm)
    linePlotter(X, Y, θ0_norm, θ1_norm, True)
    plt.scatter(X, Y, c='#ef5423', label='Data Points')

    plt.xlabel('Price')
    plt.ylabel('Mileage')
    plt.legend()
    plt.show()

    θ0, θ1 = unnormalizeΘ(X, Y, θ0_norm, θ1_norm)

    try:
        with open("../material/θ.csv", "w") as file:
            file.write(f'{ θ0 },{ θ1 }')
    except Exception as error:
        print(f'{type(error).__name__}: {error}')


if __name__ == "__main__":
    main()
