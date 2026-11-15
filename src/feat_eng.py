from typing import List
import numpy as np
import pandas as pd


def add_log1p_feat(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:

    for col in cols:

        if df[col].lt(0).mean() > 0:
            raise f'neg values in {col}'

        df[f'Log1p_{col}'] = df[col].apply(np.log1p)

    return df
