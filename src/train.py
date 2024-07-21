import numpy as np
import matplotlib.pyplot as plt

from load_csv import load
from predict import estimatePrice


def main():
    # collecting x & y
    df = load("../material/data.csv")
    X = df['km']
    Y = df['price']

    # calculate mean of x & y using an inbuilt numpy method mean()
    mean_x = np.mean(X)
    mean_y = np.mean(Y)

    # total number of input values
    m = len(X)

    # using the formula to calculate θ0 & θ1
    num = 0
    den = 0
    for i in range(m):
        num += (X[i] - mean_x) * (Y[i] - mean_y)
        den += (X[i] - mean_x) ** 2
    θ1 = num / den
    θ0 = mean_y - (θ1 * mean_x)

    # plotting values and regression line
    max_x = np.max(X) + 100
    min_x = np.min(Y) - 100

    # calculating line values x and y
    x = np.linspace(min_x, max_x, 100)
    y = θ0 + θ1 * x

    plt.plot(x, y, color='#58b970', label='Regression Line')
    plt.scatter(X, Y, c='#ef5423', label='data points')

    plt.xlabel('Price')
    plt.ylabel('Mileage')
    plt.legend()
    plt.show()

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
        with open("../material/θ.csv", "tw") as file:
            file.write(f'{ θ0 },{ θ1 }')
    except Exception as error:
        print(f'{type(error).__name__}: {error}')


if __name__ == "__main__":
    main()
