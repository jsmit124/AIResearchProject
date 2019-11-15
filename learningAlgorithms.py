#!/usr/local/bin/python3.7.4

from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

data = pd.read_csv("drug_consumption.csv")
train = data[['Alcohol', 'Cannabis', 'Cocaine', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'LSD', 'Meth', 'Mushrooms',
              'Nicotine']]
label = data[['Gender']]


def linear_regression_algorithm():
    """ @description
            Uses the linear regression algorithm to predict gender of the consituent based on the drugs they'e consumed
        @author
            Tristen, Justin
    """
    model = linear_model.LinearRegression()
    model.fit(train, label)

    print("theta_0 =", str(round(model.intercept_[0], 2)), "(intercept)")
    print("theta_1 =", str(round(model.coef_[0][0], 2)), "(Alcohol)")
    print("theta_2 =", str(round(model.coef_[0][1], 2)), "(Cannabis)")
    print("theta_3 =", str(round(model.coef_[0][2], 2)), "(Cocaine)")
    print("theta_4 =", str(round(model.coef_[0][3], 2)), "(Crack)")
    print("theta_5 =", str(round(model.coef_[0][4], 2)), "(Ecstasy)")
    print("theta_6 =", str(round(model.coef_[0][5], 2)), "(Heroin)")
    print("theta_7 =", str(round(model.coef_[0][6], 2)), "(Ketamine)")
    print("theta_8 =", str(round(model.coef_[0][7], 2)), "(LSD)")
    print("theta_9 =", str(round(model.coef_[0][8], 2)), "(Meth)")
    print("theta_10 =", str(round(model.coef_[0][9], 2)), "(Mushrooms)")
    print("theta_11 =", str(round(model.coef_[0][10], 2)), "(Nicotine)")

    instance_to_predict = np.array([6, 3, 3, 0, 4, 0, 2, 3, 0, 3, 6])
    instance_to_predict = instance_to_predict.reshape(1, -1)
    logistic_regression_prediction = model.predict(instance_to_predict)

    gender = ""

    if logistic_regression_prediction >= .5:
        gender = "Male"
    if logistic_regression_prediction < .5:
        gender = "Female"

    print("predicted y value for x =", instance_to_predict, "is", logistic_regression_prediction, "( ", gender, ")")


def decision_tree_algorithm():
    """ @description
            Uses the decision tree algorithm to predict gender of the constituent based on the drugs they'e consumed
        @author
            Justin
    """
    lb = LabelEncoder()
    data['alcohol_'] = lb.fit_transform(data['Alcohol'])
    data['cannabis_'] = lb.fit_transform(data['Cannabis'])
    data['cocaine_'] = lb.fit_transform(data['Cocaine'])
    data['crack_'] = lb.fit_transform(data['Crack'])
    data['ecstasy_'] = lb.fit_transform(data['Ecstasy'])
    data['heroin_'] = lb.fit_transform(data['Heroin'])
    data['ketamine_'] = lb.fit_transform(data['Ketamine'])
    data['lsd_'] = lb.fit_transform(data['LSD'])
    data['meth_'] = lb.fit_transform(data['Meth'])
    data['nicotine_'] = lb.fit_transform(data['Nicotine'])
    data['gender_'] = lb.fit_transform(data['Gender'])  # predict class

    print(data)

    # TODO these two lines are incorrect - need to update for use in our project - trying to figure it out (Justin)
    # x = data.iloc[:, 5:9]  # row selector, column - ':' means all rows
    # y = data.iloc[:, 9]  # row selector, column - ':' means all rows

    # tree_classifier = DecisionTreeClassifier(criterion='entropy')
    # tree_classifier.fit(x, y)

    # prediction = tree_classifier.predict(np.array([6, 3, 3, 0, 4, 0, 2, 3, 0, 3, 6]).reshape(1, -1))
    # print('Prediction to be male or female is', prediction, 'where 1 = male, 0 = female')


# def random_forest_algorithm():
""" @description
        Uses the random forest algorithm to predict gender of the constituent based on the drugs they'e consumed
    @author
        TODO
"""
# TODO


# def k_fold_cross_validation():
""" @description
        Uses k-fold cross validation to predict gender of the constituent based on the drugs they'e consumed
    @author
        TODO
"""
# TODO


def main():
    """ @description
        The main entry point for the program
    """
    linear_regression_algorithm()
    decision_tree_algorithm()
    # random_forest_algorithm()
    # k_fold_cross_validation()


main()
