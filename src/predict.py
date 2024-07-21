def estimatePrice(mileage, θ0, θ1):
    return round(θ0 + (θ1 * mileage), 2)


def main():

    try:
        try:
            with open("../material/θ.csv") as file:
                θ0, θ1 = file.read().split(',')
        except Exception as error:
            print(f'{type(error).__name__}: {error}')

        mileage = input("Enter your mileage: ")
        estimatedPrice = estimatePrice(float(mileage), float(θ0), float(θ1))
        print(estimatedPrice if estimatedPrice > 0 else 0)
    except Exception as error:
        print(f'{type(error).__name__}: {error}')


if __name__ == "__main__":
    main()
