import os
import pandas as pd
from sklearn.model_selection import train_test_split


def _read_zip(filename):
    col_names = ['id', 'event', 'device', 'channel', 'code',
                 'size', 'data']

    path = os.path.join('data', filename)

    df = pd.read_csv(path, delimiter='\t', index_col=0,
                     names=col_names,
                     compression='zip')
    n = 5000
    stop = df.channel.unique().shape[0]
    # truncate here
    return df.iloc[:n*stop, ]


# def string_to_float(string_array):
#     return list(map(float, string_array.split(',')))

def list_time_series(x):
    L = []
    for k in x.values:
        a = k.split(sep=',')
        L.append([float(z) for z in a])
    return L


def aggregate(df):
    df_listed = df.groupby(["event", "code", "device", "size"]).agg(
        data_lists=pd.NamedAgg(column='data', aggfunc=list_time_series),
        channels_names=pd.NamedAgg(column='channel', aggfunc=list)
    )
    return df_listed.reset_index().set_index('event')


# def aggregating function here
def private_public_split(df):

    df_train, df_test = train_test_split(df,
                                         test_size=0.3,
                                         random_state=42)

    df_public = df_train
    df_public_train, df_public_test = train_test_split(
                    df_public, test_size=0.2, random_state=42)

    return df_train, df_test, df_public_train, df_public_test


def to_csv_file(df, name):

    if 'public' in name:
        path = os.path.join(public_path, name)
    else:
        path = os.path.join(private_path, name)
    df.to_csv(path + '.csv')


# run prepare_data.py if you want to save the files
if __name__ == '__main__':
    public_path = os.path.join('data', 'public')
    private_path = os.path.join('data', 'private')

    # comment if you already done this
    # os.mkdir(public_path) 
    # os.mkdir(private_path)

    print("commencing preparing data, this make take a few minutes..")
    file_map = {
                "in": "original_insight.zip",
                "ep": "original_epoc.zip",
                "mw": "original_mindWave.zip",
                "mu": "original_muse.zip"
               }

    type_list = ["train", "test", "public_train", "public_test"]
    for device in file_map.keys():

        print("preparing data for '{}' device ..".format(device))
        df = _read_zip(file_map[device])
        df = aggregate(df)
        for i, x in enumerate(private_public_split(df)):
            to_csv_file(x, device + "_" + type_list[i])
