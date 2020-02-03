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
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")

data = pd.read_csv("../sortedData/ClassedDrugData.csv")


train_data = data[
    ['Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore',
     'Impulsive', 'SS']]

label = data[['ClassIllegal']]


def logistic_regression_algorithm(k, train, label):
    """ @description
            Uses the logistic regression algorithm to predict gender of the consituent based on the drugs they'e consumed
        @author
            Tristen, Justin
    """
    model = LogisticRegression(solver='saga', penalty='l1', C=1)

    score = k_fold_cross_validation(model, k, train, label)

    return score


def decision_tree_algorithm(k, train, label):
    """ @description
            Uses the decision tree algorithm to predict gender of the constituent based on the drugs they've consumed
        @author
            Justin
    """
    tree_classifier = DecisionTreeClassifier(criterion='gini', max_depth=8, max_leaf_nodes=50)

    score = k_fold_cross_validation(tree_classifier, k, train, label)

    return score


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

    return score


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

    return score


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

    return score


def k_fold_cross_validation(model, k, train, label):
    """
    @description
        Uses k-fold cross validation to test the accuracy of the models
    @author
        Aaron Merrell
    :param model: The model being scored.
    :return: the average score.
    """
    scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score),
               'recall': make_scorer(recall_score),
               'f1_score': make_scorer(f1_score)}
    folds = StratifiedKFold(n_splits=k)
    results = model_selection.cross_validate(estimator=model,
                                             X=train,
                                             y=label,
                                             cv=folds,
                                             scoring=scoring)
    return results


def get_average(array):
    sum = 0
    for i in array:
        sum += i
    return sum / float(len(array))


def main():
    """ @description
        The main entry point for the program
    """
    k_scores = queue.Queue()
    k = [2, 3, 5, 7, 9, 10, 15, 20, 50, 100]
    header = ['models', 2, 'precision', 'recall', 'f1_score', 3, 'precision', 'recall',
              'f1_score', 5, 'precision', 'recall', 'f1_score', 7, 'precision', 'recall',
              'f1_score', 9, 'precision', 'recall', 'f1_score', 10, 'precision', 'recall',
              'f1_score', 15, 'precision', 'recall', 'f1_score', 20, 'precision', 'recall',
              'f1_score', 50, 'precision', 'recall', 'f1_score', 100, 'precision', 'recall',
              'f1_score']
    k_scores.put(header)

    print("")
    print("LOGISTIC REGRESSION", sep="---")
    print("MSE Train")
    scores = ['LOGISTIC REGRESSION']
    for i in k:
        print(i)
        result = logistic_regression_algorithm(i, train_data, label)
        accuracy = result['test_accuracy']
        precision = result['test_precision']
        recall = result['test_recall']
        f1_score = result['test_f1_score']
        avg_accuracy = get_average(accuracy)
        avg_precision = get_average(precision)
        avg_recall = get_average(recall)
        avg_f1 = get_average(f1_score)
        scores.append(avg_accuracy)
        scores.append(avg_precision)
        scores.append(avg_recall)
        scores.append(avg_f1)
    k_scores.put(scores)

    print("")
    print("DECISION TREE", sep="---")
    print("MSE Train")
    scores = ['DECISION TREE']
    for i in k:
        print(i)
        result = decision_tree_algorithm(i, train_data, label)
        accuracy = result['test_accuracy']
        precision = result['test_precision']
        recall = result['test_recall']
        f1_score = result['test_f1_score']
        avg_accuracy = get_average(accuracy)
        avg_precision = get_average(precision)
        avg_recall = get_average(recall)
        avg_f1 = get_average(f1_score)
        scores.append(avg_accuracy)
        scores.append(avg_precision)
        scores.append(avg_recall)
        scores.append(avg_f1)
    k_scores.put(scores)

    print("")
    print("RANDOM FOREST", sep="---")
    print("MSE Train")
    scores = ['RANDOM FOREST']
    for i in k:
        print(i)
        result = random_forest_algorithm(i, train_data, label)
        accuracy = result['test_accuracy']
        precision = result['test_precision']
        recall = result['test_recall']
        f1_score = result['test_f1_score']
        avg_accuracy = get_average(accuracy)
        avg_precision = get_average(precision)
        avg_recall = get_average(recall)
        avg_f1 = get_average(f1_score)
        scores.append(avg_accuracy)
        scores.append(avg_precision)
        scores.append(avg_recall)
        scores.append(avg_f1)
    k_scores.put(scores)

    print("")
    print("KNN", sep="---")
    print("MSE Train")
    scores = ['KNN']
    for i in k:
        print(i)
        result = knn_algorithm(i, train_data, label)
        accuracy = result['test_accuracy']
        precision = result['test_precision']
        recall = result['test_recall']
        f1_score = result['test_f1_score']
        avg_accuracy = get_average(accuracy)
        avg_precision = get_average(precision)
        avg_recall = get_average(recall)
        avg_f1 = get_average(f1_score)
        scores.append(avg_accuracy)
        scores.append(avg_precision)
        scores.append(avg_recall)
        scores.append(avg_f1)
    k_scores.put(scores)

    print("")
    print("LINEAR DISCRIMINANT", sep="---")
    print("MSE Train")
    scores = ['LINEAR DISCRIMINANT']
    for i in k:
        print(i)
        result = linear_discriminant_algorithm(i, train_data, label)
        accuracy = result['test_accuracy']
        precision = result['test_precision']
        recall = result['test_recall']
        f1_score = result['test_f1_score']
        avg_accuracy = get_average(accuracy)
        avg_precision = get_average(precision)
        avg_recall = get_average(recall)
        avg_f1 = get_average(f1_score)
        scores.append(avg_accuracy)
        scores.append(avg_precision)
        scores.append(avg_recall)
        scores.append(avg_f1)
    k_scores.put(scores)

    
    with open('../kFoldResults/crossValidationClassIllegalResults.csv', mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=',', )
        for data in range(0, k_scores.qsize()):
            writer.writerow(k_scores.get())
    file.close()


main()
