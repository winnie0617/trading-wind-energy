import pandas as pd

# def combine_csv(first, second):


def interpolate(path_to_csv, header=0, names=None):
    df = pd.read_csv(path_to_csv, header=header, names=names)
    df.set_index('Time', inplace=True)
    df.index = pd.to_datetime(df.index)
    df = df.reindex(pd.date_range(
        start=df.index[0], end=df.index[-1], freq='H'))
    df = df.interpolate()
    return df

# def get_average_speed():
