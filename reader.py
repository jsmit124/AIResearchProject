from sklearn import linear_model
import pandas as pd
import numpy as np

data = pd.read_csv("drug_consumption.csv")
train = data[['Alcohol','Cannabis','Cocaine','Crack','Ecstasy','Heroin','Ketamine','LSD','Meth','Mushrooms','Nicotine']]
label = data[['Gender']]

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
prediction = model.predict(instance_to_predict)

gender = ""

if(prediction >= .5):
         gender = "Male"
if(prediction < .5):
         gender = "Female"
         

print("predicted y value for x =", instance_to_predict, "is", prediction, "( ", gender, ")")
