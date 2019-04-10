import pandas as pd
import numpy as np
from os import path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

PATH = path.dirname(path.realpath(__file__))
DATA_FILENAME = "ElectionsData.csv"
DATA_PATH = PATH + "/" + DATA_FILENAME


def load_data(filepath):
    df = pd.read_csv(filepath, header=0)
    return df


def partionner(data_frame, test_size, validation_size):
    validation_size = validation_size / (1 - test_size)
    train_set_temp, test_set = train_test_split(data_frame, test_size=test_size)
    train_set, validation_set = train_test_split(train_set_temp, test_size=validation_size)
    return train_set, validation_set, test_set


def categorize_data(df):
    object_columns = df.keys()[df.dtypes.map(lambda x: x == 'object')]
    for curr_column in object_columns:
        df[curr_column] = df[curr_column].astype("category")
        df[curr_column] = df[curr_column].cat.rename_categories(range(df[curr_column].nunique())).astype(int)
        df.loc[df[curr_column].isnull(), curr_column] = np.nan  # fix NaN conversion


def negative_2_NaN(df):
    df[df < 0] = np.nan  # fix NaN conversion


def data_cleansing(df):
    pass


def main():
    df = load_data(DATA_PATH)
    categorize_data(df)
    negative_2_NaN(df)
    df.to_csv(PATH + "/temp.csv")
    # plt.hist(df[].values)
    # plt.show()
    # categorize_data(df)
    # df.to_csv(path_or_buf=PATH + "/temp.csv")
    # partionner(df, 0.25, 0.25)
    # train_set, validation_set, test_set = partionner(df, 0.25, 0.25)
    # train_set_raw, validation_set_raw, test_set_raw = train_set.copy(), validation_set.copy(), test_set.copy()
    #
    # return train_set_raw, validation_set_raw, test_set_raw


if __name__ == '__main__':
    main()
