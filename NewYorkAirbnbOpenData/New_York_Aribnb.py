# made 2022/5//18 14:00
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def print_test(msg):
    print(f"hello{msg}")


# Define function StandardScaler + Label encoding
def Standard_Label(numerical_features, categorical_features):
    # Numerical_features(StandardScaler)
    # Dependent variables
    col_to_scale = numerical_features.columns

    # Independent variable(Target value)
    y = numerical_features.loc[:, ['price']]
    numerical_features = numerical_features.drop(['price'], axis=1)
    col_to_scale = numerical_features.columns
    numerical_features[col_to_scale] = StandardScaler().fit_transform(numerical_features[col_to_scale])

    # Categorical feature(Label encoding)
    label = LabelEncoder()
    col_to_encode = categorical_features.columns
    categorical_features[col_to_encode] = label.fit_transform(categorical_features[col_to_encode])

    # pandas dataset X
    X = pd.concat([numerical_features, categorical_features], axis=1)

    Processed_sc_label = pd.concat([X, y], axis=1)

    return X, y, Processed_sc_label


# Define function StandardScaler + OneHot encoding
def Standard_OneHot(numerical_features, categorical_features):
    # Numerical_features(StandardScaler)
    # Dependent variables
    col_to_scale = numerical_features.columns

    # Independent variable(Target value)
    y = numerical_features.loc[:, ['price']]
    numerical_features = numerical_features.drop(['price'], axis=1)
    col_to_scale = numerical_features.columns
    numerical_features[col_to_scale] = StandardScaler().fit_transform(numerical_features[col_to_scale])

    # Categorical feature(OneHot encoding)
    # Using pandas.get_dummies
    categorical_features = pd.get_dummies(categorical_features, columns=categorical_features.columns)

    # pandas dataset X
    X = pd.concat([numerical_features, categorical_features], axis=1)

    Processed_sc_oneHot = pd.concat([X, y], axis=1)

    return X, y, Processed_sc_oneHot


# Define function RobustScaler + Label encoding
def Robust_Label(numerical_features, categorical_features):
    # Numerical_features(RobustScaler)
    # Dependent variables
    col_to_scale = numerical_features.columns

    # Independent variable(Target value)
    y = numerical_features.loc[:, ['price']]
    numerical_features = numerical_features.drop(['price'], axis=1)
    col_to_scale = numerical_features.columns
    numerical_features[col_to_scale] = RobustScaler().fit_transform(numerical_features[col_to_scale])

    # Categorical feature(Label encoding)
    label = LabelEncoder()
    col_to_encode = categorical_features.columns
    categorical_features[col_to_encode] = label.fit_transform(categorical_features[col_to_encode])

    # pandas dataset X
    X = pd.concat([numerical_features, categorical_features], axis=1)

    Processed_R_label = pd.concat([X, y], axis=1)

    return X, y, Processed_R_label


# Define function RobustScaler + OneHot encoding
def Robust_OneHot(numerical_features, categorical_features):
    # Numerical_features(RobustScaler)
    # Dependent variables
    col_to_scale = numerical_features.columns

    # Independent variable(Target value)
    y = numerical_features.loc[:, ['price']]
    numerical_features = numerical_features.drop(['price'], axis=1)
    numerical_features[col_to_scale] = RobustScaler().fit_transform(numerical_features[col_to_scale])

    # Categorical feature(OneHot encoding)
    # Using pandas.get_dummies
    categorical_features = pd.get_dummies(categorical_features, columns=categorical_features.columns)

    # pandas dataset X
    X = pd.concat([numerical_features, categorical_features], axis=1)

    Processed_R_oneHot = pd.concat([X, y], axis=1)

    return X, y, Processed_R_oneHot


# Define function MinMaxScaler + Label encoding
def MinMax_Label(numerical_features, categorical_features):
    # Numerical_features(MinMaxScaler)
    # Dependent variables
    col_to_scale = numerical_features.columns

    # Independent variable(Target value)
    y = numerical_features.loc[:, ['price']]
    numerical_features = numerical_features.drop(['price'], axis=1)
    col_to_scale = numerical_features.columns
    numerical_features[col_to_scale] = MinMaxScaler().fit_transform(numerical_features[col_to_scale])

    # Categorical feature(Label encoding)
    label = LabelEncoder()
    col_to_encode = categorical_features.columns
    categorical_features[col_to_encode] = label.fit_transform(categorical_features[col_to_encode])

    # pandas dataset X
    X = pd.concat([numerical_features, categorical_features], axis=1)

    Processed_MinMax_label = pd.concat([X, y], axis=1)

    return X, y, Processed_MinMax_label


# Define function MinMaxScaler + OneHot encoding
def MinMax_OneHot(numerical_features, categorical_features):
    # Numerical_features(MinMaxScaler)
    # Dependent variables
    col_to_scale = numerical_features.columns

    # Independent variable(Target value)
    y = numerical_features.loc[:, ['price']]
    numerical_features = numerical_features.drop(['price'], axis=1)
    numerical_features[col_to_scale] = MinMaxScaler().fit_transform(numerical_features[col_to_scale])

    # Categorical feature(OneHot encoding)
    # Using pandas.get_dummies
    categorical_features = pd.get_dummies(categorical_features, columns=categorical_features.columns)

    # pandas dataset X
    X = pd.concat([numerical_features, categorical_features], axis=1)

    Processed_MinMax_oneHot = pd.concat([X, y], axis=1)

    return X, y, Processed_MinMax_oneHot


# train-test split and print automatically
def print_train_test(scaling_encoding_x, scaling_encoding_y, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(scaling_encoding_x, scaling_encoding_y, test_size=test_size,
                                                        random_state=random_state)
    print('Dimensions of the training feature matrix: {}'.format(X_train.shape))
    print('Dimensions of the training target vector: {}'.format(y_train.shape))
    print('Dimensions of the test feature matrix: {}'.format(X_test.shape))
    print('Dimensions of the test target vector: {}'.format(y_test.shape))
    return X_train, X_test, y_train, y_test
