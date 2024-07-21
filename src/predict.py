def estimatePrice(mileage, θ0, θ1):
    return θ0 + (θ1 * mileage)


def main():

    θ0 = θ1 = 0

    try:
        try:
            with open("../material/θ.csv") as file:
                θ0, θ1 = file.read().split(',')
        except Exception:
            print('The model has not been trained.')

        mileage = input("Enter your mileage: ")
        estimatedPrice = round(estimatePrice(float(mileage), float(θ0),
                                             float(θ1)), 2)
        print(estimatedPrice if estimatedPrice > 0 else 0)
    except Exception as error:
        print(f'{type(error).__name__}: {error}')


if __name__ == "__main__":
    main()
