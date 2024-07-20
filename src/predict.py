def estimatePrice(mileage, θ0, θ1):
    return θ0 + (θ1 * mileage)


def main():
    θ0 = θ1 = 0

    try:
        try:
            with open("../material/θ.csv") as file:
                θ0, θ1 = file.read().split(',')
        except Exception as error:
            print(f'{type(error).__name__}: {error}')

        mileage = input("Enter your mileage: ")
        print(estimatePrice(float(mileage), float(θ0), float(θ1)))
    except Exception as error:
        print(f'{type(error).__name__}: {error}')


if __name__ == "__main__":
    main()
