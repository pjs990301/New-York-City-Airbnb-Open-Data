# made 2022/5//18 14:00
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.tree import DecisionTreeRegressor


def print_test(msg):
    print(f"hello{msg}")


# Define function StandardScaler + Label encoding
def Standard_Label(numerical_features, categorical_features):
    # Numerical_features(StandardScaler)
    # Independent variable(Target value)
    y = numerical_features.loc[:, ['price']]

    # Dependent variables
    numerical_features = numerical_features.drop(['price'], axis=1)
    col_to_scale = numerical_features.columns
    numerical_features[col_to_scale] = StandardScaler().fit_transform(numerical_features[col_to_scale])

    # Categorical feature(Label encoding)
    label = LabelEncoder()
    col_to_encode = categorical_features.columns
    for i in range(len(categorical_features.columns)):
        categorical_features[col_to_encode[i]] = label.fit_transform(categorical_features[col_to_encode[i]])

    # pandas dataset X
    X = pd.concat([numerical_features, categorical_features], axis=1)

    Processed_sc_label = pd.concat([X, y], axis=1)

    return X, y, Processed_sc_label


# Define function StandardScaler + OneHot encoding
def Standard_OneHot(numerical_features, categorical_features):
    # Numerical_features(StandardScaler)
    # Independent variable(Target value)
    y = numerical_features.loc[:, ['price']]

    # Dependent variables
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
    # Independent variable(Target value)
    y = numerical_features.loc[:, ['price']]

    # Dependent variables
    numerical_features = numerical_features.drop(['price'], axis=1)
    col_to_scale = numerical_features.columns
    numerical_features[col_to_scale] = RobustScaler().fit_transform(numerical_features[col_to_scale])

    # Categorical feature(Label encoding)
    label = LabelEncoder()
    col_to_encode = categorical_features.columns
    for i in range(len(categorical_features.columns)):
        categorical_features[col_to_encode[i]] = label.fit_transform(categorical_features[col_to_encode[i]])

    # pandas dataset X
    X = pd.concat([numerical_features, categorical_features], axis=1)

    Processed_R_label = pd.concat([X, y], axis=1)

    return X, y, Processed_R_label


# Define function RobustScaler + OneHot encoding
def Robust_OneHot(numerical_features, categorical_features):
    # Numerical_features(RobustScaler)
    # Independent variable(Target value)
    y = numerical_features.loc[:, ['price']]

    # Dependent variables
    numerical_features = numerical_features.drop(['price'], axis=1)
    col_to_scale = numerical_features.columns
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
    # Independent variable(Target value)
    y = numerical_features.loc[:, ['price']]

    # Dependent variables
    numerical_features = numerical_features.drop(['price'], axis=1)
    col_to_scale = numerical_features.columns
    numerical_features[col_to_scale] = MinMaxScaler().fit_transform(numerical_features[col_to_scale])

    # Categorical feature(Label encoding)
    label = LabelEncoder()
    col_to_encode = categorical_features.columns
    for i in range(len(categorical_features.columns)):
        categorical_features[col_to_encode[i]] = label.fit_transform(categorical_features[col_to_encode[i]])

    # pandas dataset X
    X = pd.concat([numerical_features, categorical_features], axis=1)

    Processed_MinMax_label = pd.concat([X, y], axis=1)

    return X, y, Processed_MinMax_label


# Define function MinMaxScaler + OneHot encoding
def MinMax_OneHot(numerical_features, categorical_features):
    # Numerical_features(MinMaxScaler)
    # Independent variable(Target value)
    y = numerical_features.loc[:, ['price']]

    # Dependent variables
    numerical_features = numerical_features.drop(['price'], axis=1)
    col_to_scale = numerical_features.columns
    numerical_features[col_to_scale] = MinMaxScaler().fit_transform(numerical_features[col_to_scale])

    # Categorical feature(OneHot encoding)
    # Using pandas.get_dummies
    categorical_features = pd.get_dummies(categorical_features, columns=categorical_features.columns)

    # pandas dataset X
    X = pd.concat([numerical_features, categorical_features], axis=1)

    Processed_MinMax_oneHot = pd.concat([X, y], axis=1)

    return X, y, Processed_MinMax_oneHot


def SE_LinearRegression(X_train, X_test, y_train, y_test):
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    score = r2_score(y_test, y_pred)
    MAE = mean_absolute_error(y_test, y_pred)
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    MAPE = mean_absolute_percentage_error(y_test, y_pred)
    return score, MAE, RMSE, MAPE


def SE_RandomForestRegressor(X_train, X_test, y_train, y_test):
    forest_model = RandomForestRegressor()
    forest_model.fit(X_train, y_train)
    y_pred = forest_model.predict(X_test)
    score = r2_score(y_test, y_pred)
    MAE = mean_absolute_error(y_test, y_pred)
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    MAPE = mean_absolute_percentage_error(y_test, y_pred)
    return score, MAE, RMSE, MAPE


def SE_DecisionTreeRegressor(X_train, X_test, y_train, y_test):
    DTree = DecisionTreeRegressor()
    DTree.fit(X_train, y_train)
    y_pred = DTree.predict(X_test)
    score = r2_score(y_test, y_pred)
    MAE = mean_absolute_error(y_test, y_pred)
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    MAPE = mean_absolute_percentage_error(y_test, y_pred)
    return score, MAE, RMSE, MAPE


def SE_KNeighborsClassifier(X_train, X_test, y_train, y_test):
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)
    train_score = classifier.score(X_train, y_train)
    test_score = classifier.score(X_test, y_test)

    k_list = range(1, 101)
    accuracies = []
    for k in k_list:
        classifier = KNeighborsClassifier(n_neighbors=k)
        classifier.fit(X_train, y_train)
        accuracies.append(classifier.score(X_test, y_test))

    plt.figure(figsize=(7, 7))
    plt.plot(k_list, accuracies)
    plt.xlabel("k")
    plt.ylabel("Validation Accuracy")
    plt.title("Breast Cancer Classifier Accuracy")
    plt.show()

    return train_score, test_score


def SE_ExtraTreesClassifier(X_train, X_test, y_train, y_test):
    feature_model = ExtraTreesClassifier(n_estimators=10)
    feature_model.fit(X_train, y_train)
    train_score = feature_model.score(X_train, y_train)
    test_score = feature_model.score(X_test, y_test)
    plt.figure(figsize=(7, 7))
    feat_importance = pd.Series(feature_model.feature_importances_, index=X_test.columns)
    feat_importance.nlargest(20).plot(kind='barh')
    plt.show()
    return train_score, test_score


# train-test split and print automatically
def print_train_test(scaling_encoding_x, scaling_encoding_y, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(scaling_encoding_x, scaling_encoding_y, test_size=test_size,
                                                        random_state=random_state)
    print('Dimensions of the training feature matrix: {}'.format(X_train.shape))
    print('Dimensions of the training target vector: {}'.format(y_train.shape))
    print('Dimensions of the test feature matrix: {}'.format(X_test.shape))
    print('Dimensions of the test target vector: {}'.format(y_test.shape))
    return X_train, X_test, y_train, y_test
