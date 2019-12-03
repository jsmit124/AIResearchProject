#!/usr/local/bin/python3.7.4
"""
@description
    This script takes a data set with drug use and uses 'Age', 'Education', 'Country', 'Ethnicity', 'Alcohol',
    'Cannabis', 'Cocaine', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'LSD', 'Meth', 'Mushrooms',
    'Nicotine' to predict gender. The data is run through seven models and the accuracy score of
    each model is printed to the console.
"""
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics

warnings.filterwarnings("ignore")

# data = pd.read_csv("drug_consumption.csv")
test_data = pd.read_csv("test_data.csv")
train_data = pd.read_csv("train_data.csv")
validation_data = pd.read_csv("validation_data.csv")

test_train = test_data[['Age', 'Education', 'Country', 'Ethnicity', 'Alcohol', 'Cannabis', 'Cocaine', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'LSD', 'Meth', 'Mushrooms',
                   'Nicotine']]
train_train = train_data[['Age', 'Education', 'Country', 'Ethnicity', 'Alcohol', 'Cannabis', 'Cocaine', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'LSD', 'Meth', 'Mushrooms',
                    'Nicotine']]
validation_train = validation_data[['Age', 'Education', 'Country', 'Ethnicity', 'Alcohol', 'Cannabis', 'Cocaine', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'LSD', 'Meth', 'Mushrooms',
                         'Nicotine']]

test_label = test_data[['Gender']]
train_label = train_data[['Gender']]
validation_label = validation_data[['Gender']]

# train = data[['Age', 'Education', 'Country', 'Ethnicity', 'Alcohol', 'Cannabis', 'Cocaine', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'LSD', 'Meth', 'Mushrooms',
#               'Nicotine']]
# label = data[['Gender']]


def logistic_regression_algorithm(k, train, label):
    """ @description
            Uses the logistic regression algorithm to predict gender of the consituent based on the drugs they'e consumed
        @author
            Tristen, Justin
    """
    model = LogisticRegression(solver='liblinear', penalty='l2', C=0.001)
    # model.fit(train, label)

    score = k_fold_cross_validation(model, k, train, label)

    print('Linear Regression accuracy is: {}%'.format(round(score, 1)))


def decision_tree_algorithm(k, train, label):
    """ @description
            Uses the decision tree algorithm to predict gender of the constituent based on the drugs they've consumed
        @author
            Justin
    """
    # lb = LabelEncoder()
    # data['age_'] = lb.fit_transform(data['Age'])
    # data['education_'] = lb.fit_transform(data['Education'])
    # data['country_'] = lb.fit_transform(data['Country'])
    # data['ethnicity_'] = lb.fit_transform(data['Ethnicity'])
    # data['alcohol_'] = lb.fit_transform(data['Alcohol'])
    # data['cannabis_'] = lb.fit_transform(data['Cannabis'])
    # data['cocaine_'] = lb.fit_transform(data['Cocaine'])
    # data['crack_'] = lb.fit_transform(data['Crack'])
    # data['ecstasy_'] = lb.fit_transform(data['Ecstasy'])
    # data['heroin_'] = lb.fit_transform(data['Heroin'])
    # data['ketamine_'] = lb.fit_transform(data['Ketamine'])
    # data['lsd_'] = lb.fit_transform(data['LSD'])
    # data['meth_'] = lb.fit_transform(data['Meth'])
    # data['mushrooms_'] = lb.fit_transform(data['Mushrooms'])
    # data['nicotine_'] = lb.fit_transform(data['Nicotine'])
    # data['gender_'] = lb.fit_transform(data['Gender'])  # predict class

    # x = data.iloc[:, [1, 2, 3, 4, 12, 17, 19, 20, 21, 22, 23, 25, 26, 27, 28]]
    # y = data.iloc[:, 31]

    tree_classifier = DecisionTreeClassifier(criterion='gini', max_depth=3, max_leaf_nodes=10)
    # tree_classifier.fit(x, y)

    score = k_fold_cross_validation(tree_classifier, k, train, label)

    print('Decision Tree accuracy is: {}%'.format(round(score, 1)))


def random_forest_algorithm(k, train, label):
    """
     @description
        Uses the random forest algorithm to predict gender of the constituent based on the drugs they've consumed
     @author
        Aaron Merrell
     :return: The accuracy of the model.
    """
    classifier = RandomForestClassifier(criterion='gini', max_depth=4, n_estimators=10)
    # the_label = np.ravel(label)
    # classifier.fit(train, the_label)

    score = k_fold_cross_validation(classifier, k, train, label)

    print('Random forest accuracy is: {}%'.format(round(score, 1)))


def knn_algorithm(k, train, label):
    """
    @description
        uses the SVC model to predict the gender of the constituent based on the drugs they've consumed.
    @author
        Aaron Merrell
    :return: The accuracy of the model.
    """
    knn = KNeighborsClassifier(algorithm='ball_tree', leaf_size=6, metric='minkowski', n_neighbors=25, p=2, weights='uniform')
    # the_label = np.ravel(label)
    # knn.fit(train, the_label)

    score = k_fold_cross_validation(knn, k, train, label)

    print('knn accuracy is: {}%'.format(round(score, 1)))


def linear_discriminant_algorithm(k, train, label):
    """
    @description
        uses the SVC model to predict the gender of the constituent based on the drugs they've consumed.
    @author
        Aaron Merrell
    :return: The accuracy of the model.
    """
    lda = LinearDiscriminantAnalysis(solver='svd')
    # the_label = np.ravel(label)
    # lda.fit(train, the_label)

    score = k_fold_cross_validation(lda, k, train, label)

    print('Linear Discriminant accuracy is: {}%'.format(round(score, 1)))



def k_fold_cross_validation(model, k, train, label):
    """
    @description
        Uses k-fold cross validation to test the accuracy of the models
    @author
        Aaron Merrell
    :param model: The model being scored.
    :return: the average score.
    """
    scores = []
    folds = StratifiedKFold(n_splits=k)
    for train_index, test_index in folds.split(train, label):
        x_train, x_test, y_train, y_test = train.iloc[train_index], train.iloc[test_index], label.iloc[train_index], label.iloc[test_index]
        scores.append(get_score(model, x_train, x_test, y_train, y_test))

    avg = 0
    for i in scores:
        avg += i
    avg = (avg / len(scores)) * 100
    return avg


def get_score(model, x_train, x_test, y_train, y_test):
    """
    @description
        Gets the accuracy score for the madel
    @author
        Aaron Merrell
    @:parameter
    :param model: The model to test
    :param x_train: The features to train model by
    :param x_test: The features to test model by
    :param y_train: The label to train the model by
    :param y_test: The label to test the model by
    :return: The accuracy of the model
    """
    model.fit(x_train, np.ravel(y_train))
    return model.score(x_test, y_test)


def grid_search_logistic():
    """
    @description
        Finds the best parameters for the logistic algorithm.
    @author
        Aaron Merrell
    :return: A dictionary with the best parameters.
    """
    model = LogisticRegression()
    param_grid = [{'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'penalty': ['l2', 'none']},
                  {'solver': ['liblinear', 'saga'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'penalty': ['l1', 'l2']}]
    clf = GridSearchCV(model, param_grid=param_grid, cv=5, n_jobs=1)
    the_label = np.ravel(train_label)
    best_clf = clf.fit(train_train, the_label)
    print(best_clf.best_params_)

def grid_search_decision():
    """
    @description
        Finds the best parameters for the decision tree algorithm.
    @author
        Aaron Merrell
    :return: A dictionary with the best parameters.
    """
    model = DecisionTreeClassifier()
    param_grid = {'criterion': ['gini', 'entropy'], 'max_depth': np.arange(1, 51), 'max_leaf_nodes': [5, 10, 20, 50, 100]}
    clf = GridSearchCV(model, param_grid=param_grid, cv=5, n_jobs=1)
    the_label = np.ravel(train_label)
    best_clf = clf.fit(train_train, the_label)
    print(best_clf.best_params_)

def grid_search_forest():
    """
    @description
        Finds the best parameters for the random forest algorithm.
    @author
        Aaron Merrell
    :return: A dictionary with the best parameters.
    """
    model = RandomForestClassifier()
    param_grid = { 'criterion': ['gini'], 'max_depth': [3, 4, 5, 6], 'n_estimators': np.arange(1, 1001, 10)}
    clf = GridSearchCV(model, param_grid=param_grid, cv=5, n_jobs=1)
    the_label = np.ravel(train_label)
    best_clf = clf.fit(train_train, the_label)
    print(best_clf.best_params_)


def grid_search_knn():
    """
    @description
        Finds the best parameters for the knn algorithm.
    @author
        Aaron Merrell
    :return: A dictionary with the best parameters.
    """
    model = KNeighborsClassifier()
    param_grid = { 'n_neighbors': np.arange(1, 51), 'weights': ['uniform', 'distance'],
                   'algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto'], 'leaf_size': np.arange(1, 51),
                   'p': [1, 2], 'metric': ['minkowski']}
    clf = GridSearchCV(model, param_grid=param_grid, cv=5, n_jobs=1)
    the_label = np.ravel(train_label)
    best_clf = clf.fit(train_train, the_label)
    print(best_clf.best_params_)


def grid_search_linear_discriminant():
    """
    @description
        Finds the best parameters for the linear discriminant algorithm.
    @author
        Aaron Merrell
    :return: A dictionary with the best parameters.
    """
    model = LinearDiscriminantAnalysis()
    param_grid = { 'solver': ['svd', 'lsqr', 'eigen']}
    clf = GridSearchCV(model, param_grid=param_grid, cv=5, n_jobs=1)
    the_label = np.ravel(train_label)
    best_clf = clf.fit(train_train, the_label)
    print(best_clf.best_params_)

def ada_boost(train, label):
    """
    @description
        Gets the accuracy score for the ada boost model
    @author
        Tristen Rivera
    @:parameter
    :param train: The model to test
    :param label: The label to test the model by
    :return: The accuracy of the model
    """
    
    X_train, X_test, y_train, y_test = train_test_split(train, label, test_size=0.3)

    # Create adaboost classifer object
    abc = AdaBoostClassifier(n_estimators=50, learning_rate=1)

    # Train Adaboost Classifer
    model = abc.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = model.predict(X_test)

    print("Ada Boost accuracy is: {}%".format(round(metrics.accuracy_score(y_test, y_pred) * 100.0,1)))


def main():
    """ @description
        The main entry point for the program
    """
    k = [2, 3, 5, 7, 9, 10, 15, 20, 50, 100]

    print("ADA BOOST", sep="---")
    print("MSE Train")
    for i in k:
        print(i)
        ada_boost(train_train, train_label)
    print("MSE Validation")
    for i in k:
        print(i)
        ada_boost(validation_train, validation_label)
    print("MSE Test")
    for i in k:
        print(i)
        ada_boost(test_train, test_label)
        
    print("LOGISTIC REGRESSION", sep="---")
    print("MSE Train")
    for i in k:
        print(i)
        logistic_regression_algorithm(i, train_train, train_label)
    print("MSE Validation")
    for i in k:
        print(i)
        logistic_regression_algorithm(i, validation_train, validation_label)
    print("MSE Test")
    for i in k:
        print(i)
        logistic_regression_algorithm(i, test_train, test_label)

    print("")
    print("DECISION TREE", sep="---")
    print("MSE Train")
    for i in k:
        print(i)
        decision_tree_algorithm(i, train_train, train_label)
    print("MSE Validation")
    for i in k:
        print(i)
        decision_tree_algorithm(i, validation_train, validation_label)
    print("MSE Test")
    for i in k:
        print(i)
        decision_tree_algorithm(i, test_train, test_label)

    print("")
    print("RANDOM FOREST", sep="---")
    print("MSE Train")
    for i in k:
        print(i)
        random_forest_algorithm(i, train_train, train_label)
    print("MSE Validation")
    for i in k:
        print(i)
        random_forest_algorithm(i, validation_train, validation_label)
    print("MSE Test")
    for i in k:
        print(i)
        random_forest_algorithm(i, test_train, test_label)

    print("")
    print("KNN", sep="---")
    print("MSE Train")
    for i in k:
        print(i)
        knn_algorithm(i, train_train, train_label)
    print("MSE Validation")
    for i in k:
        print(i)
        knn_algorithm(i, validation_train, validation_label)
    print("MSE Test")
    for i in k:
        print(i)
        knn_algorithm(i, test_train, test_label)

    print("")
    print("LINEAR DISCRIMINANT", sep="---")
    print("MSE Train")
    for i in k:
        print(i)
        linear_discriminant_algorithm(i, train_train, train_label)
    print("MSE Validation")
    for i in k:
        print(i)
        linear_discriminant_algorithm(i, validation_train, validation_label)
    print("MSE Test")
    for i in k:
        print(i)
        linear_discriminant_algorithm(i, test_train, test_label)


main()


# folds                 2      3      5      7      9       10     15     20     50     100
# logistic regression  65.5   65.6   65.5   65.6   65.8    65.9   65.9   66.0   65.9   66.0
# decision tree        61.4   65.0   64.7   65.8   64.7    65.0   64.4   65.7   64.8   65.5
# random forest        65.3   64.6   64.9   65.5   65.9    65.7   65.6   66.0   66.4   65.8
# knn                  63.6   64.1   65.3   65.0   65.2    65.2   65.1   64.8   64.9   65.1
# linear_discriminant  64.2   64.3   64.5   65.1   65.2    65.3   65.5   65.3   65.4   65.6


# hyperparameters tuned with grid search
# logistic regression {'C': 0.001, 'penalty': l2, 'solver': liblinear}
# decision tree {'criterion': gini, 'max_depth': 3, 'max_leaf_nodes': 10}
# random forest {'criterion: 'gini', 'max_depth': 4, 'n_estimators': 10}
# knn {'algorithm': 'ball_tree', 'leaf_size': 6, 'metric': 'minkowski', 'n_neighbors': 25, 'p': 2, 'weights': 'uniform'}
# linear_discriminant {'solver': 'svd'}
