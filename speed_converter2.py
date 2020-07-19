import pandas as pd
from preprocess import interpolate


def combine_speeds(path1, path2):
    '''
    Takes the paths of the historical and predicted data of one farm, drops the direction column, and returns a dataframe object.
    '''
    df1 = pd.read_csv(path1, header=3)
    df2 = pd.read_csv(path2, header=3)
    df_combined = pd.concat([df1, df2])
    return df_combined.drop(columns='Speed(m/s)')


if __name__ == "__main__":
    csv_list = [['AppendixData/angerville-1.csv', 'AppendixData/angerville-1-b.csv'], ['AppendixData/angerville-2.csv',
                                                                                       'AppendixData/angerville-2-b.csv'], ['AppendixData/arville.csv', 'AppendixData/arville-b.csv'], ['AppendixData/boissy-la-riviere.csv', 'AppendixData/boissy-la-riviere-b.csv'], ['AppendixData/guitrancourt.csv', 'AppendixData/guitrancourt-b.csv'], ['AppendixData/lieusaint.csv', 'AppendixData/lieusaint-b.csv'], ['AppendixData/lvs-pussay.csv', 'AppendixData/lvs-pussay-b.csv'], ['AppendixData/parc-du-gatinais.csv', 'AppendixData/parc-du-gatinais-b.csv']]
    df_list = []
    for pair in csv_list:
        df = combine_speeds(pair[0], pair[1])
        df = interpolate(df)
        df_list.append(df)
    df = pd.concat(df_list, axis=1)
    df.to_csv('all-directions-speed.csv', index_label='Time', header=['Speed (m/s)', 'Speed (m/s)', 'Speed (m/s)', 'Speed (m/s)', 'Speed (m/s)', 'Speed (m/s)', 'Speed (m/s)', 'Speed (m/s)'])

