import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import sklearn

data = pd.read_pickle('./baseline/name_sal_pts.pkl')
x = data['college_ppg'].values
y = data['salary'].values
x = x.reshape(len(data.index), 1)
y = y.reshape(len(data.index), 1)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)


regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)
pred_train = regr.predict(x_train)
pred_test = regr.predict(x_test)
print("Training error (MSE):", np.mean((y_train - pred_train)**2))
print("Test error (MSE):", np.mean((y_test - pred_test)**2))
print("Training error (MAE):", np.mean(abs(y_train - pred_train)))
print("Test error (MAE):", np.mean(abs(y_test - pred_test)))


plt.scatter(x_train, y_train, color='blue', marker='.', linewidths=0.2, alpha=0.5)
plt.scatter(x_test, pred_test, color='green', marker='.', linewidths=0.2, alpha=0.5)

line, = plt.plot(x, regr.predict(x), color='black', linewidth=2.0)
line.set_antialiased(True)
plt.xlabel("College points per game")
plt.ylabel("Salary ($1M USD)")
plt.show()




