import numpy as np
import pandas as pd
import errno
import sys
import os


def main():
    np.set_printoptions(threshold=np.inf)  # prints a full matrix rather than an abbreviated matrix
    # pd.options.display.max_colwidth = 100   # number of characters of lyrics shown when dataframe is printed

    print("\n\t##### EXECUTING CLEAN_DATA.PY #####\n")

    dest_path = '../Data/data/'

    # read in arguments from command line
    try:
        data_set_path = str(sys.argv[1])
    except (ValueError, IndexError):
        print('Arguments must be specified as follows:')
        print('python3 Clean_Data.py <path-to-csv>')
        sys.exit(0)

    if os.path.exists(data_set_path):
        df = pd.read_csv(data_set_path, dtype=str, header=0)
    else:
        print('File not found; path does not exist')
        sys.exit(0)

    print('Extracting relevant information ...')
    if '2014' in data_set_path:
        features = [1, 3, 4, 5, 6, 7, 9, 10, 12, 15, 14, 11]
        data_set_idx = [8, 9, 10, 11]
        year = '2014'
    elif '2015' in data_set_path:
        features = [1, 4, 5, 6, 7, 8, 10, 14, 15, 17, 20, 19, 16]
        data_set_idx = [9, 10, 11, 12]
        year = '2015'
    elif '2016' in data_set_path:
        features = [1, 4, 5, 6, 7, 8, 10, 14, 15, 17, 20, 19, 16]
        data_set_idx = [9, 10, 11, 12]
        year = '2016'
    elif '2017' in data_set_path:
        features = [1, 4, 5, 6, 7, 8, 10, 14, 15, 17, 20, 19, 16]
        data_set_idx = [9, 10, 11, 12]
        year = '2017'
    elif '2018' in data_set_path:
        features = [0, 1, 2, 3, 4, 5, 6, 7, 9, 12, 11, 8]
        data_set_idx = [8, 9, 10, 11]
        year = '2018'
    else:
        print('Provided incorrect year. Must be years 2014-2018')
        sys.exit()

    df = df[[
        'collectionSiteId',
        'lake',
        'algalBloom',
        'algalBloomSheen',
        'batherLoad',
        'correct_timestamp',
        'turbidity',
        'waterAppearance',
        'waterfowlPresence',
        'waterTemp',
        'waveIntensity',
        'lat',
        'long'
    ]]

    df = df.dropna(axis=0, how='any')

    # cast data types
    for col in df.columns:
        if col not in ['correct_timestamp', 'collectionSiteId', 'lake']:
            df[col] = df[col].astype(float)
        elif col == 'correct_timestamp':
            df['correct_timestamp'] = pd.to_datetime(df['correct_timestamp'])

    # rename columns
    df.rename(
        {'correct_timestamp': 'date', 'lat': 'latitude', 'long': 'longitude', 'collectionSiteId': 'site'},
        axis='columns',
        inplace=True
    )

    # variable transformations
    df['log_turbidity'] = np.log(df['turbidity'] + 1)   # log transform turbidity
    df['cos_hour'] = np.zeros(shape=len(df))
    df['sin_hour'] = np.zeros(shape=len(df))

    print(df.loc[3, 'date'].hour + (df.loc[3, 'date'].minute / 60))

    for i in range(len(df)):
        time_num = df.loc[i, 'date'].hour + (df.loc[i, 'date'].minute / 60)
        df.loc[i, 'cos_hour'] = np.cos(2 * np.pi * (time_num / 24))
        df.loc[i, 'sin_hour'] = np.sin(2 * np.pi * (time_num / 24))

    # df['cos_hour'] = np.cos(2 * np.pi * ())

    # keep features for prediction or sorting
    df = df[[
        'date',
        'algalBloomSheen',
        'lake',
        'site',
        'turbidity',
        'log_turbidity',
        'waterTemp',
        'waveIntensity',
        'latitude',
        'longitude'
    ]]


    # todo split df into smaller df's based on site, and then order based on datetime, and then back and front fill
    # todo log transform some variables!!!


if __name__ == "__main__":
    main()
