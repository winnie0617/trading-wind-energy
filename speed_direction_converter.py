import csv
import numpy as np
import pandas as pd

def speed_direction_converter(path_to_csv):
    df = pd.read_csv(path_to_csv, header=3)
    df.set_index('Time', inplace=True)
    df.index = pd.to_datetime(df.index)
    df = df.reindex(pd.date_range(start=df.index[0], end=df.index[-1], freq='H'))
    df = df.interpolate()