import gzip
import numpy as np
import os
import pandas as pd
import requests
import shutil

'''
For details and discussion around the Snake Eyes dataset see the folloing:

https://github.com/nlw0/snake-eyes/tree/master
https://www.kaggle.com/datasets/nicw102168/snake-eyes
https://www.kaggle.com/code/nicw102168/a-stroll-through-the-neighborhood-manifold
'''

base = 'https://raw.githubusercontent.com/nlw0/snake-eyes/master/'

files = [
    'snakeeyes_00.dat',
    'snakeeyes_01.dat',
    'snakeeyes_02.dat',
    'snakeeyes_03.dat',
    'snakeeyes_04.dat',
    'snakeeyes_05.dat',
    'snakeeyes_06.dat',
    'snakeeyes_07.dat',
    'snakeeyes_08.dat',
    'snakeeyes_09.dat',
]


def download_snake_eyes_data():

    for file_name in files:
        print(f'downloading: {file_name}.gz...')
        r = requests.get(f'{base}{file_name}.gz')
        with open(f'examples/{file_name}.gz', 'wb') as test_file:
            test_file.write(r.content)


def extract_all_ones():
    for file_name in files:
        if not os.path.isfile(f'examples/{file_name}.gz'):
            download_snake_eyes_data()

    df_list = []
    for file_name in files:
        print(f'extracting data from: {file_name}')

        with gzip.open(f'examples/{file_name}.gz', 'rb') as raw_data:
            data = np.frombuffer(raw_data.read(), np.uint8)
            data = data.reshape(100000, 401)

        df = pd.DataFrame(data)

        df_list.append(df)

    print('merging all data...')
    df_all = pd.concat(df_list)

    print('filtering ones only...')
    df_ones = df_all[df_all[0] == 1]
    df_ones = df_ones.drop(columns=[0])

    print(f'saving {len(df_ones)} rows to: examples/snake_eyes_ones.csv')
    df_ones.to_csv('examples/snake_eyes_ones.csv', index=False)

    print('saving compressed data: examples/snake_eyes_ones.csv.gz')
    with open('examples/snake_eyes_ones.csv', 'rb') as f_in:
        with gzip.open('examples/snake_eyes_ones.csv.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def load_smooth_snake_eyes_ones():
    '''
    Returns the ones only from the Snake Eyes data as a list of tuples
    length 400.

    This format can be used to create a EuclideanSubspace object i.e.

    points = load_snake_eyes_ones()
    space = EuclideanSubspace(points)

    if the file is not present in the examples subfolder the raw
    data is downloaded from the original source.
    '''
    if os.path.isfile('examples/snake_eyes_ones_smooth.csv'):
        df = pd.read_csv('examples/snake_eyes_ones_smooth.csv')
    # elif os.path.isfile('examples/snake_eyes_ones.csv.gz'):
    #     df = pd.read_csv(
    #         'examples/snake_eyes_ones.csv.gz',
    #         compression='gzip'
    #     )

    #     print(f'saving {len(df)} rows to: examples/snake_eyes_ones.csv')
    #     df.to_csv('examples/snake_eyes_ones.csv', index=False)
    # else:
    #     print('cannot locate file: examples/snake_eyes_ones.csv')
    #     extract_all_ones()
    #     df = pd.read_csv('examples/snake_eyes_ones.csv')

    return list(df.itertuples(index=False, name=None))


def load_snake_eyes_ones():
    '''
    Returns the ones only from the Snake Eyes data as a list of tuples
    length 400.

    This format can be used to create a EuclideanSubspace object i.e.

    points = load_snake_eyes_ones()
    space = EuclideanSubspace(points)

    if the file is not present in the examples subfolder the raw
    data is downloaded from the original source.
    '''
    if os.path.isfile('examples/snake_eyes_ones.csv'):
        df = pd.read_csv('examples/snake_eyes_ones.csv')
    elif os.path.isfile('examples/snake_eyes_ones.csv.gz'):
        df = pd.read_csv(
            'examples/snake_eyes_ones.csv.gz',
            compression='gzip'
        )

        print(f'saving {len(df)} rows to: examples/snake_eyes_ones.csv')
        df.to_csv('examples/snake_eyes_ones.csv', index=False)
    else:
        print('cannot locate file: examples/snake_eyes_ones.csv')
        extract_all_ones()
        df = pd.read_csv('examples/snake_eyes_ones.csv')

    return list(df.itertuples(index=False, name=None))
