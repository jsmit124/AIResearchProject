#!/usr/local/bin/python3.7.4

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

data = pd.read_csv("drug_consumption.csv")

train = data[['Age', 'Education', 'Country', 'Ethnicity', 'Alcohol', 'Cannabis', 'Cocaine', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'LSD', 'Meth', 'Mushrooms',
              'Nicotine']]
label = data[['Gender']]


def logistic_regression_algorithm():
    """ @description
            Uses the logistic regression algorithm to predict gender of the consituent based on the drugs they'e consumed
        @author
            Tristen, Justin
    """
    model = LogisticRegression(solver='liblinear', penalty='l2', C=0.001)
    # model.fit(train, label)

    score = k_fold_cross_validation(model)

    print('Linear Regression accuracy is: {}%'.format(round(score, 1)))


def decision_tree_algorithm():
    """ @description
            Uses the decision tree algorithm to predict gender of the constituent based on the drugs they've consumed
        @author
            Justin
    """
    lb = LabelEncoder()
    data['age_'] = lb.fit_transform(data['Age'])
    data['education_'] = lb.fit_transform(data['Education'])
    data['country_'] = lb.fit_transform(data['Country'])
    data['ethnicity_'] = lb.fit_transform(data['Ethnicity'])
    data['alcohol_'] = lb.fit_transform(data['Alcohol'])
    data['cannabis_'] = lb.fit_transform(data['Cannabis'])
    data['cocaine_'] = lb.fit_transform(data['Cocaine'])
    data['crack_'] = lb.fit_transform(data['Crack'])
    data['ecstasy_'] = lb.fit_transform(data['Ecstasy'])
    data['heroin_'] = lb.fit_transform(data['Heroin'])
    data['ketamine_'] = lb.fit_transform(data['Ketamine'])
    data['lsd_'] = lb.fit_transform(data['LSD'])
    data['meth_'] = lb.fit_transform(data['Meth'])
    data['mushrooms_'] = lb.fit_transform(data['Mushrooms'])
    data['nicotine_'] = lb.fit_transform(data['Nicotine'])
    data['gender_'] = lb.fit_transform(data['Gender'])  # predict class

    x = data.iloc[:, [1, 2, 3, 4, 12, 17, 19, 20, 21, 22, 23, 25, 26, 27, 28]]
    y = data.iloc[:, 31]

    tree_classifier = DecisionTreeClassifier(criterion='gini', max_depth=3, max_leaf_nodes=10)
    # tree_classifier.fit(x, y)

    score = k_fold_cross_validation(tree_classifier)

    print('Decision Tree accuracy is: {}%'.format(round(score, 1)))


def random_forest_algorithm():
    """ @description
            Uses the random forest algorithm to predict gender of the constituent based on the drugs they've consumed
        @author
            Aaron Merrell
    """
    classifier = RandomForestClassifier(criterion='gini', max_depth=4, n_estimators=10)
    the_label = np.ravel(label)
    # classifier.fit(train, the_label)

    score = k_fold_cross_validation(classifier)

    print('Random forest accuracy is: {}%'.format(round(score, 1)))


def k_fold_cross_validation(model):
    """ @description
            Uses k-fold cross validation to test the accuracy of the models
        @author
            Aaron Merrell
    """
    scores = []
    #TODO play with k
    folds = StratifiedKFold(n_splits=10)
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
    model = LogisticRegression()
    param_grid = [{'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'penalty': ['l2', 'none']},
                  {'solver': ['liblinear', 'saga'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'penalty': ['l1', 'l2']}]
    clf = GridSearchCV(model, param_grid=param_grid, cv=5, n_jobs=1)
    the_label = np.ravel(label)
    best_clf = clf.fit(train, the_label)
    print(best_clf.best_params_)

def grid_search_decision():
    model = DecisionTreeClassifier()
    param_grid = {'criterion': ['gini', 'entropy'], 'max_depth': np.arange(1, 51), 'max_leaf_nodes': [5, 10, 20, 50, 100]}
    clf = GridSearchCV(model, param_grid=param_grid, cv=5, n_jobs=1)
    the_label = np.ravel(label)
    best_clf = clf.fit(train, the_label)
    print(best_clf.best_params_)

def grid_search_forest():
    model = RandomForestClassifier()
    param_grid = { 'criterion': ['gini', 'entropy'], 'max_depth': np.arange(1, 51), 'n_estimators': [10, 30, 50, 70, 90, 100, 300, 500, 700, 900, 1000]}
    clf = GridSearchCV(model, param_grid=param_grid, cv=5, n_jobs=1)
    the_label = np.ravel(label)
    best_clf = clf.fit(train, the_label)
    print(best_clf.best_params_)


def main():
    """ @description
        The main entry point for the program
    """
    print("LOGISTIC REGRESSION", sep="---")
    logistic_regression_algorithm()
    # grid_search_logistic()
    print("")
    print("DECISION TREE", sep="---")
    decision_tree_algorithm()
    # grid_search_decision()
    print("")
    print("RANDOM FOREST", sep="---")
    random_forest_algorithm()
    # grid_search_forest()


main()


# folds                 2      3      5      7      9       10     15     20     50     100
# logistic regression  65.5   65.6   65.5   65.6   65.8    65.9   65.9   66.0   65.9   66.0
# decision tree        61.4   65.0   64.7   65.8   64.7    65.0   64.4   65.7   64.8   65.5
# random forest        65.3   64.6   64.9   65.5   65.9    65.7   65.6   66.0   66.4   65.8

# hyperparameters tuned with grid search
# logistic regression {'C': 0.001, 'penalty': l2, 'solver': liblinear}
# decision tree {'criterion': gini, 'max_depth': 3, 'max_leaf_nodes': 10}
# random forest {'criterion: 'gini', 'max_depth': 4, 'n_estimators': 10}
