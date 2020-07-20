import pandas as pd


# def interpolate(path_to_csv, header=0, names=None):
#     df = pd.read_csv(path_to_csv, header=header, names=names)
#     df.set_index('Time', inplace=True)
#     df.index = pd.to_datetime(df.index)
#     df = df.reindex(pd.date_range(
#         start=df.index[0], end=df.index[-1], freq='H'))
#     df = df.interpolate()
#     return df

def interpolate(df):
    df.set_index('Time', inplace=True)
    df.index = pd.to_datetime(df.index)
    df = df.reindex(pd.date_range(
        start=df.index[0], end=df.index[-1], freq='H'))
    df = df.interpolate()
    return df


def get_average_speed(csv_list):
    data_list = []
    for csv in csv_list:
        df = pd.read_csv('AppendixData/'+csv, header=3)
        df = interpolate(df)
        print(df.shape)
        data_list.append(df.drop(columns='Direction (deg N)'))
    df = pd.concat(data_list, axis=1).mean(axis=1)
    df.to_csv('average-wind-speed.csv', index_label='Time',
              header=['Average Speed (m/s)'])
    print("Updated average speed")

def get_average_direction(csv_list):
    data_list = []
    for csv in csv_list:
        df = pd.read_csv('AppendixData/'+csv, header=3)
        df = interpolate(df)
        print(df.shape)
        data_list.append(df.drop(columns='Speed(m/s)'))
    df = pd.concat(data_list, axis=1).mean(axis=1)
    df.to_csv('average-wind-direction.csv', index_label='Time',
              header=['Average Direction (deg N)'])
    print("Updated average direction")