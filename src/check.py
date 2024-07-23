import numpy as np

from load_csv import load
from predict import estimatePrice


def main():
    # collecting X and Y
    try:
        df = load("../material/data.csv")
    except Exception:
        print('The database could not be retrieved.')
        return 1

    X = df['km']
    Y = df['price']

    # calculate mean of Y
    mean_y = np.mean(Y)

    # initialising θ0 and θ1
    try:
        with open("../material/θ.csv") as file:
            θ0, θ1 = file.read().split(',')
    except Exception:
        print('The model has not been trained.')
        return 1

    # calculating R-squared value for measuring the goodness of our model

    try:
        ss_t = 0  # total sum of squares
        ss_r = 0  # total sum of square of residuals

        for i in range(int(len(X))):
            y_pred = estimatePrice(X[i], float(θ0), float(θ1))
            ss_t += (Y[i] - mean_y) ** 2
            ss_r += (Y[i] - y_pred) ** 2
        r2 = 1 - (ss_r/ss_t)

        print(r2)
    except Exception as error:
        print(f'{type(error).__name__}: {error}')


if __name__ == "__main__":
    main()
