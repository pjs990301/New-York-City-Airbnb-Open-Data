import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def print_test(msg) :
    print(f"hello{msg}")

# Define function StandardScaler + Label encoding
def Standard_Label(numerical_features, categorical_features):
    # Numerical_features(StandardScaler)
    # Dependent variables
    col_to_scale = ['host_id', 'latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'reviews_per_month',
                    'calculated_host_listings_count', 'availability_365']

    # Independent variable(Target value)
    y = numerical_features.loc[:, ['price']]
    numerical_features = numerical_features.drop(['price'], axis=1)
    numerical_features[col_to_scale] = StandardScaler().fit_transform(numerical_features[col_to_scale])

    # Categorical feature(Label encoding)
    label = LabelEncoder()
    categorical_features['neighbourhood'] = label.fit_transform(categorical_features['neighbourhood'])
    categorical_features['room_type'] = label.fit_transform(categorical_features['room_type'])
    categorical_features['neighbourhood_group'] = label.fit_transform(categorical_features['neighbourhood_group'])

    # pandas dataset X
    X = pd.concat([numerical_features, categorical_features], axis=1)

    Processed_sc_label = pd.concat([X, y], axis=1)

    return X, y, Processed_sc_label


# Define function StandardScaler + OneHot encoding
def Standard_OneHot(numerical_features, categorical_features):
    # Numerical_features(StandardScaler)
    # Dependent variables
    col_to_scale = ['host_id', 'latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'reviews_per_month',
                    'calculated_host_listings_count', 'availability_365']

    # Independent variable(Target value)
    y = numerical_features.loc[:, ['price']]
    numerical_features = numerical_features.drop(['price'], axis=1)
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
    col_to_scale = ['host_id', 'latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'reviews_per_month',
                    'calculated_host_listings_count', 'availability_365']

    # Independent variable(Target value)
    y = numerical_features.loc[:, ['price']]
    numerical_features = numerical_features.drop(['price'], axis=1)
    numerical_features[col_to_scale] = RobustScaler().fit_transform(numerical_features[col_to_scale])

    # Categorical feature(Label encoding)
    label = LabelEncoder()
    categorical_features['neighbourhood'] = label.fit_transform(categorical_features['neighbourhood'])
    categorical_features['room_type'] = label.fit_transform(categorical_features['room_type'])
    categorical_features['neighbourhood_group'] = label.fit_transform(categorical_features['neighbourhood_group'])

    # pandas dataset X
    X = pd.concat([numerical_features, categorical_features], axis=1)

    Processed_R_label = pd.concat([X, y], axis=1)

    return X, y, Processed_R_label


# Define function RobustScaler + OneHot encoding
def Robust_OneHot(numerical_features, categorical_features):
    # Numerical_features(RobustScaler)
    # Dependent variables
    col_to_scale = ['host_id', 'latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'reviews_per_month',
                    'calculated_host_listings_count', 'availability_365']

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
    col_to_scale = ['host_id', 'latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'reviews_per_month',
                    'calculated_host_listings_count', 'availability_365']

    # Independent variable(Target value)
    y = numerical_features.loc[:, ['price']]
    numerical_features = numerical_features.drop(['price'], axis=1)
    numerical_features[col_to_scale] = MinMaxScaler().fit_transform(numerical_features[col_to_scale])

    # Categorical feature(Label encoding)
    label = LabelEncoder()
    categorical_features['neighbourhood'] = label.fit_transform(categorical_features['neighbourhood'])
    categorical_features['room_type'] = label.fit_transform(categorical_features['room_type'])
    categorical_features['neighbourhood_group'] = label.fit_transform(categorical_features['neighbourhood_group'])

    # pandas dataset X
    X = pd.concat([numerical_features, categorical_features], axis=1)

    Processed_MinMax_label = pd.concat([X, y], axis=1)

    return X, y, Processed_MinMax_label


# Define function MinMaxScaler + OneHot encoding
def MinMax_OneHot(numerical_features, categorical_features):
    # Numerical_features(MinMaxScaler)
    # Dependent variables
    col_to_scale = ['host_id', 'latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'reviews_per_month',
                    'calculated_host_listings_count', 'availability_365']

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