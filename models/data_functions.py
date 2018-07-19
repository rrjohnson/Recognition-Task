"""
Various functions for dealing with the data
"""

import numpy as np
import pandas as pd


def generate_mean_dfs(dataframes):

    mean_dfs = {}

    for key in dataframes.keys():
        mean_dfs[key] = pd.DataFrame(columns=[*dataframes[list(dataframes.keys())[0]].columns])

    for name, df in dataframes.items():

        conversations = df.reset_index()['conversation'].unique()
        # std deviation code
#         mean_dfs['std_n_words'] = None
#         mean_dfs['std_duration'] = None
#         mean_dfs['std_gap'] = None

        for conversation in conversations:
            update = df.loc[conversation][[i for i in df.columns if i != 'type']].apply(np.mean)
            update['type'] = df.loc[conversation].iloc[0]['type']
            # Code to include standard deviation
#             update['std_n_words'] = df.loc[conversation]['n_words'].describe()['std']
#             update['std_gap'] = df.loc[conversation]['gap'].describe()['std']
#             update['std_duration'] = df.loc[conversation]['duration'].describe()['std']
            mean_dfs[name] = mean_dfs[name].append(update, ignore_index=True)

        mean_dfs[name].loc[mean_dfs[name]['type'] == 'good', ['type']] = 1
        mean_dfs[name].loc[mean_dfs[name]['type'] == 'bad', ['type']] = 0
        mean_dfs[name].reset_index(inplace=True)
        mean_dfs[name].drop(['index'], axis=1, inplace=True)

    return mean_dfs

def generate_test_training(dataframes):
    """
    Generate test and training data
    ------------------------------
    Input:
        dataframes: dict of 2 dataframes (test and training)

    Output:
        X, y - Full datasets
        X_train, y_train - Training
        X_test, y_test - Testing
    """

    # Get full set and test training
    train_test_full = pd.DataFrame(columns=[*list(dataframes.values())[0].columns])
    train_test_full = train_test_full.append(dataframes['train'])
    train_test_full = train_test_full.append(dataframes['test'])

    X = train_test_full.drop(['type'], axis=1)
    y = train_test_full['type'].reset_index()
    y.drop(['index'], axis=1, inplace=True)
    y = list(map(int, y['type'].values))

    X_train = dataframes['train'].drop(['type'], axis=1)
    y_train = list(map(int, dataframes['train']['type'].values))
    X_test = dataframes['test'].drop(['type'], axis=1)
    y_test = list(map(int, dataframes['test']['type'].values))

    return X, y, X_train, y_train, X_test, y_test

