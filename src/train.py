import numpy as np
import matplotlib.pyplot as plt
import argparse

from load_csv import load
from predict import estimatePrice


def linePlotter(X, Y, θ0_norm, θ1_norm, index=0):
    # Transforming θ0 and θ1 back to original scale
    θ0, θ1 = unnormalizeΘ(X, Y, θ0_norm, θ1_norm)

    # plotting values and regression line
    max_x = np.max(X) + 100
    min_x = np.min(X) - 100

    # calculating line values x and y
    x = np.linspace(min_x, max_x, 100)
    y = θ0 + θ1 * x

    if not index:
        plt.plot(x, y, label='Regression Line')
    else:
        plt.plot(x, y, label=f'Regression Line step { index }')


def gradientDescent(X, Y, θ0, θ1, learningRate, iterations, printInterval):
    m = len(X)  # Number of population values

    for i in range(iterations):
        sum_tmpθ0 = 0.0
        sum_tmpθ1 = 0.0

        for j in range(m):
            error = estimatePrice(X[j], θ0, θ1) - Y[j]
            sum_tmpθ0 += error
            sum_tmpθ1 += error * X[j]

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
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--graph", dest="graph",
                        action=argparse.BooleanOptionalAction,
                        help="It displays the graph with the regression line \
                            and the data points")
    parser.add_argument("-sS", "--showStages",
                        action=argparse.BooleanOptionalAction,
                        dest="showStages",
                        help="It shows regression line stages")

    args = parser.parse_args()

    # collecting X and Y
    try:
        df = load("../material/data.csv")
    except Exception:
        print('The database could not be retrieved.')
        return 1

    X = df['km']
    Y = df['price']

    # using the formula to calculate θ0 and θ1
    θ0 = θ1 = 1
    learningRate = 0.01
    iterations = 10000

    i = 1
    # I normalize my arrays
    normalizedX = (X - X.mean()) / X.std()
    normalizedY = (Y - Y.mean()) / Y.std()

    for θ0_norm, θ1_norm in gradientDescent(normalizedX, normalizedY, θ0, θ1,
                                            learningRate, iterations, 50):
        θ0, θ1 = unnormalizeΘ(X, Y, θ0_norm, θ1_norm)
        if args.showStages:
            linePlotter(X, Y, θ0_norm, θ1_norm, i)
            i += 1
    if args.graph or args.showStages:
        linePlotter(X, Y, θ0_norm, θ1_norm)
        plt.scatter(X, Y, c='#ef5423', label='Data Points')

        plt.xlabel('Mileage')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    try:
        with open("../material/θ.csv", "w") as file:
            file.write(f'{ θ0 },{ θ1 }')
    except Exception as error:
        print(f'{type(error).__name__}: {error}')


if __name__ == "__main__":
    main()
