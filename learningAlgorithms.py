#!/usr/local/bin/python3.7.4
"""
@description
    This script takes a data set with drug use and uses 'Age', 'Education', 'Country', 'Ethnicity', 'Alcohol',
    'Cannabis', 'Cocaine', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'LSD', 'Meth', 'Mushrooms',
    'Nicotine' to predict gender. The data is run through seven models and the accuracy score of
    each model is printed to the console.
"""
import csv
import queue
import warnings

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

train_data = pd.read_csv("sortedData/ClassedDrugData.csv")
train_train = train_data[
    ['Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore',
     'Impulsive', 'SS', 'Class00', 'Class10', 'Class01']]
train_label = train_data[['Class11']]


def logistic_regression_algorithm(k, train, label):
    """ @description
            Uses the logistic regression algorithm to predict gender of the consituent based on the drugs they'e consumed
        @author
            Tristen, Justin
    """
    model = LogisticRegression(solver='saga', penalty='l1', C=1)

    score = k_fold_cross_validation(model, k, train, label)

    print('Linear Regression accuracy is: {}%'.format(round(score, 1)))
    return round(score, 1)


def decision_tree_algorithm(k, train, label):
    """ @description
            Uses the decision tree algorithm to predict gender of the constituent based on the drugs they've consumed
        @author
            Justin
    """
    tree_classifier = DecisionTreeClassifier(criterion='gini', max_depth=8, max_leaf_nodes=50)

    score = k_fold_cross_validation(tree_classifier, k, train, label)

    print('Decision Tree accuracy is: {}%'.format(round(score, 1)))
    return round(score, 1)


def random_forest_algorithm(k, train, label):
    """
     @description
        Uses the random forest algorithm to predict gender of the constituent based on the drugs they've consumed
     @author
        Aaron Merrell
     :return: The accuracy of the model.
    """
    classifier = RandomForestClassifier(criterion='gini', max_depth=5, n_estimators=31)

    score = k_fold_cross_validation(classifier, k, train, label)

    print('Random forest accuracy is: {}%'.format(round(score, 1)))
    return round(score, 1)


def knn_algorithm(k, train, label):
    """
    @description
        uses the SVC model to predict the gender of the constituent based on the drugs they've consumed.
    @author
        Aaron Merrell
    :return: The accuracy of the model.
    """
    knn = KNeighborsClassifier(algorithm='ball_tree', leaf_size=1, metric='minkowski', n_neighbors=19, p=2,
                               weights='distance')

    score = k_fold_cross_validation(knn, k, train, label)

    print('knn accuracy is: {}%'.format(round(score, 1)))
    return round(score, 1)


def linear_discriminant_algorithm(k, train, label):
    """
    @description
        uses the SVC model to predict the gender of the constituent based on the drugs they've consumed.
    @author
        Aaron Merrell
    :return: The accuracy of the model.
    """
    lda = LinearDiscriminantAnalysis(solver='svd')

    score = k_fold_cross_validation(lda, k, train, label)

    print('Linear Discriminant accuracy is: {}%'.format(round(score, 1)))
    return round(score, 1)


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
        x_train, x_test, y_train, y_test = train.iloc[train_index], train.iloc[test_index], label.iloc[train_index], \
                                           label.iloc[test_index]
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


def main():
    """ @description
        The main entry point for the program
    """
    k_scores = queue.Queue()
    k = [2, 3, 5, 7, 9, 10, 15, 20, 50, 100]
    header = ['models', 2, 3, 5, 7, 9, 10, 15, 20, 50, 100]
    k_scores.put(header)

    print("")
    print("LOGISTIC REGRESSION", sep="---")
    print("MSE Train")
    scores = ['LOGISTIC REGRESSION']
    for i in k:
        print(i)
        scores.append(logistic_regression_algorithm(i, train_train, train_label))
    k_scores.put(scores)

    print("")
    print("DECISION TREE", sep="---")
    print("MSE Train")
    scores = ['DECISION TREE']
    for i in k:
        print(i)
        scores.append(decision_tree_algorithm(i, train_train, train_label))
    k_scores.put(scores)

    print("")
    print("RANDOM FOREST", sep="---")
    print("MSE Train")
    scores = ['RANDOM FOREST']
    for i in k:
        print(i)
        scores.append(random_forest_algorithm(i, train_train, train_label))
    k_scores.put(scores)

    print("")
    print("KNN", sep="---")
    print("MSE Train")
    scores = ['KNN']
    for i in k:
        print(i)
        scores.append(knn_algorithm(i, train_train, train_label))
    k_scores.put(scores)

    print("")
    print("LINEAR DISCRIMINANT", sep="---")
    print("MSE Train")
    scores = ['LINEAR DISCRIMINANT']
    for i in k:
        print(i)
        scores.append(linear_discriminant_algorithm(i, train_train, train_label))
    k_scores.put(scores)

    with open('./kFoldResults/crossValidationClass11Results.csv', mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=',', )
        for data in range(0, k_scores.qsize()):
            writer.writerow(k_scores.get())
    file.close()


main()

# folds                 2      3      5      7      9       10     15     20     50     100
# ada boost            77.4   78.4   80.4   79.0   79.2    80.2   78.6   77.7   79.5   78.8
# logistic regression  79.0   78.3   79.1   79.6   79.5    79.6   79.9   80.1   80.2   80.3
# decision tree        74.8   75.8   77.2   75.9   76.7    76.7   75.9   77.7   77.5   77.0
# random forest        78.1   76.9   78.5   78.7   79.7    79.3   79.1   79.4   79.3   79.9
# knn                  78.7   77.5   79.2   79.9   79.3    79.5   79.8   80.1   80.1   79.9
# linear_discriminant  78.5   76.8   78.4   79.1   79.1    79.0   79.5   79.6   79.8   79.9


# hyperparameters tuned with grid search
# logistic regression {'C': 1, 'penalty': l1, 'solver': saga}
# decision tree {'criterion': gini, 'max_depth': 8, 'max_leaf_nodes': 50}
# random forest {'criterion: 'gini', 'max_depth': 5, 'n_estimators': 31}
# knn {'algorithm': 'ball_tree', 'leaf_size': 1, 'metric': 'minkowski', 'n_neighbors': 19, 'p': 2, 'weights': 'distance'}
# linear_discriminant {'solver': 'svd'}
