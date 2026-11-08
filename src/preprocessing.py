from typing import Tuple

import pandas as pd


def train_valid_test_split(df: pd.DataFrame, cutoff: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cut = int(len(df) * cutoff)
    train = df.iloc[:cut]
    valid = df.iloc[cut:int(cut + (1-cutoff)/2)]
    test = df.iloc[int(cut + (1-cutoff)/2):]
    return train, valid, test