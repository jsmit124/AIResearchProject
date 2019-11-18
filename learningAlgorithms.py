#!/usr/local/bin/python3.7.4


from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
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

    score = k_fold_cross_validation(model)

    print("Predicted y value for x =", instance_to_predict, "is", logistic_regression_prediction, "(", gender, ")")
    print('Linear Regression accuracy is: {}%'.format(round(score, 1)))


def decision_tree_algorithm():
    """ @description
            Uses the decision tree algorithm to predict gender of the constituent based on the drugs they've consumed
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
    data['mushrooms_'] = lb.fit_transform(data['Mushrooms'])
    data['nicotine_'] = lb.fit_transform(data['Nicotine'])
    data['gender_'] = lb.fit_transform(data['Gender'])  # predict class

    x = data.iloc[:, [12, 17, 19, 20, 21, 22, 23, 25, 26, 27, 28]]
    y = data.iloc[:, 31]

    tree_classifier = DecisionTreeClassifier(criterion='entropy')
    tree_classifier.fit(x, y)

    instance_to_predict = np.array([6, 3, 3, 0, 4, 0, 2, 3, 0, 3, 6])
    prediction = tree_classifier.predict(instance_to_predict.reshape(1, -1))
    score = k_fold_cross_validation(tree_classifier)

    print('Prediction to be male or female is', prediction, 'where 1 = Male, 0 = Female')
    print('Decision Tree accuracy is: {}%'.format(round(score, 1)))


def random_forest_algorithm():
    """ @description
            Uses the random forest algorithm to predict gender of the constituent based on the drugs they've consumed
        @author
            Aaron Merrell
    """
    classifier = RandomForestClassifier(n_estimators=1000)
    the_label = np.ravel(label)
    classifier.fit(train, the_label)
    instance_to_predict = np.array([6, 3, 3, 0, 4, 0, 2, 3, 0, 3, 6])
    instance_to_predict = instance_to_predict.reshape(1, -1)
    y_pred = classifier.predict(instance_to_predict)

    gender = ''

    if y_pred == 1:
        gender = 'Male'
    if y_pred == 0:
        gender = 'Female'

    score = k_fold_cross_validation(classifier)

    print("Predicted y value for x =", instance_to_predict, "is", y_pred, "(", gender, ")")
    print('Random forest accuracy is: {}%'.format(round(score, 1)))


def k_fold_cross_validation(model):
    """ @description
            Uses k-fold cross validation to test the accuracy of the models
        @author
            Aaron Merrell
    """
    scores = []
    folds = KFold(n_splits=10)
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


def main():
    """ @description
        The main entry point for the program
    """
    print("LINEAR REGRESSION", sep="---")
    linear_regression_algorithm()
    print("")
    print("DECISION TREE", sep="---")
    decision_tree_algorithm()
    print("")
    print("RANDOM FOREST", sep="---")
    random_forest_algorithm()


main()
