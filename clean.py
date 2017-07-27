import pandas as pd
import numpy as np


# Obviously, reading the data is a good start
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")


# TODO: See about genericizing this for other data sets
def fill_values(data, s_cats, s_nums):
    """
    Fill all the columns with the median or mode of the column, depending on type.
    
    :param data: The data frame containing the aforementioned columns.
    :param s_cats: Categorical columns requiring special handling
    :param s_nums: Numeric columns requiring special handling
    :return: The same data frame with the values filled in with medians and modes.
    """
    numeric_types = (np.dtype('int64'), np.dtype('float64'))

    for col in data.columns.values:
        if col in special_cats:
            data[col] = data[col].fillna('NoExist')
        elif col in special_nums:
            data[col] = data[col].fillna(0)
        elif data[col].dtype in numeric_types:
            data[col] = data[col].fillna(int(data[col].median()))
        elif data[col].dtype is np.dtype('object'):
            data[col] = data[col].fillna(data[col].mode().iloc[0])

    return data


def dummy_vars(data):
    """
    Replace categorical variables with dummy variables.

    :param data: The data frame containing the aforementioned columns.
    :return: Transformed data frame with dummy variables.
    """
    new_data = None

    for col in data.columns.values:
        if data[col].dtype is np.dtype('object'):
            dummies = pd.get_dummies(data[col])
            new_data = pd.concat([data, dummies], axis='columns')
            del new_data[col]

    return new_data


# Impute special values
special_cats = ('FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
                    'GarageCond', 'PoolQC', 'Fence', 'MiscFeature')
special_nums = ('GarageYrBlt')

partial_clean_train = fill_values(train, special_cats, special_nums)
clean_train = (train.pipe(fill_values, s_cats=special_cats, s_nums=special_nums)
               .pipe(dummy_vars))
