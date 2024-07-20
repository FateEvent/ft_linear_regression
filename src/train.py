import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from load_csv import load
from predict import estimatePrice


def main():
    df = load("../material/data.csv")
    X = df['km']
    Y = df['price']

    mean_x = np.mean(X)
    mean_y = np.mean(Y)
    a = len(X)

    num = 0
    den = 0
    for i in range(a):
        num += (X[i] - mean_x) * (Y[i] - mean_y)
        den += (X[i] - mean_x) ** 2
    a = num / den
    b = mean_y - (a * mean_x)

    # plotting values and regression line
    max_x = np.max(X) + 100
    min_x = np.min(Y) - 100

    # calculating line values x and y
    x = np.linspace(min_x, max_x, 100)
    y = a * x + b

    # calculating R-squared value for measuring goodness of our model.

    ss_t = 0  # total sum of squares
    ss_r = 0  # total sum of square of residuals

    for i in range(int(len(X))):
        y_pred = a * X[i] + b
        ss_t += (Y[i] - mean_y) ** 2
        ss_r += (Y[i] - y_pred) ** 2
    r2 = 1 - (ss_r/ss_t)

    print(r2)

    plt.plot(x, y, color='#58b970', label='Regression Line')
    plt.scatter(X, Y, c='#ef5423', label='data points')

    plt.xlabel('Price')
    plt.ylabel('Mileage')
    plt.legend()
    plt.show()

    try:
        with open("../material/θ.csv", "tw") as file:
            file.write(f'{ b },{ a }')
    except Exception as error:
        print(f'{type(error).__name__}: {error}')


if __name__ == "__main__":
    main()
