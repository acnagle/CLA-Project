import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

diabetes = datasets.load_diabetes()

diabetes_X = diabetes.data[:, np.newaxis, 2]

diabetes_X_train = diabetes_X[:20]
diabetes_X_test = diabetes_X[-20:]

diabetes_y_train = diabetes.target[:20]
diabetes_y_test = diabetes.target[-20:]

regr = linear_model.LinearRegression()

regr.fit(diabetes_X_train, diabetes_y_train)

diabetes_y_pred = regr.predict(diabetes_X_test)

print("Coefficients: \n", regr.coef_)
print("MSE: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
print("Variance score: %2f" % r2_score(diabetes_y_test, diabetes_y_pred))

print("\nPredicted Labels:")
print(diabetes_y_pred)

print("\nTarget Labels:")
print(diabetes_y_test)
