import os
import pandas as pd


def load(path: str) -> pd.DataFrame:
    """The load() function loads a CSV file, prints the dimensions \
of the data set on the terminal and returns it."""

    if not os.path.isfile(path):
        raise AssertionError("invalid path")

    df = pd.read_csv(path)

    print(f'Loading dataset of dimensions { df.shape }')
    return df
