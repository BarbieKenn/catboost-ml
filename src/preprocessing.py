from typing import Tuple

import pandas as pd


def train_valid_test_split(df, cutoff=0.7):
    train_end = int(len(df) * cutoff)
    valid_end = int(len(df) * (cutoff + (1 - cutoff) / 2))

    return df.iloc[:train_end], df.iloc[train_end:valid_end], df.iloc[valid_end:]
