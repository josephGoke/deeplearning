import numpy as np
import pandas as pd


def time_series_split(X, y, train_size: float):
    '''
    Time series train_test_split function: maintains temporal order
    
    :X: Features array
    :y: Target array
    :test_size: Proportion of data for testing(0.0 to 1.0)
    :test_size: float

    '''

    n_samples = X.shape[0]

    #Split point
    train_samples = int(n_samples * train_size)

    X_train = X[:train_samples]
    X_test = X[train_samples:]
    y_train = y[:train_samples]
    y_test = y[train_samples:]

    print(f'\nTimes-series Split:')
    print(f'Training Samples: {len(X_train)} (first {train_size*100:.0f}% of data) ')
    print(f'Testing Samples: {len(X_test)} (last {(1-train_size)*100:.0f}% of data) ')

    return X_train, X_test, y_train, y_test