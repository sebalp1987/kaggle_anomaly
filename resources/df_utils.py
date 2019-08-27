import numpy as np


def clean_inf_nan(df):
    return df.replace([np.inf, -np.inf], np.nan)