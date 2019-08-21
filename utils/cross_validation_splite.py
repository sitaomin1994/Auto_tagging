import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

# split train validation and test set
def split_train_val_test(df):
    '''
    Split train validation test set
    :param df: dataframe of data
    :return: train, val, test dataframe
    '''

    train, validate, test = np.split(df.sample(frac=1), [int(.6 * len(df)), int(.8 * len(df))])

    return train, validate, test

# cross-validation
def generate_cross_validation(training_data_X, training_data_y, n_fold = 5):

    '''
    :param training_data_X - X_train
    :param training_data_y - y_train
    :param n_fold - cv_fold
    :return: cv_data - A List of (X_train, X_test, y_train, y_test) for cross_validation
    '''

    kf = KFold(n_splits=2)
    cv_data = []
    for train_index, test_index in kf.split(training_data_X):
        X_train, X_test = training_data_X[train_index], training_data_X[test_index]
        y_train, y_test = training_data_y[train_index], training_data_y[test_index]
        cv_data.append((X_train, X_test, y_train, y_test))

    return cv_data


