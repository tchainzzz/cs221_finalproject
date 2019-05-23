import csv
from sklearn import svm
import pandas
from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 

colnamesX = ['Player','PTS_x', 'DRPG_x', 'TOPG_x', 'PF_x', 'RPG_x', 'Position', 'Percentile']
colnamesY = ['Class']
dataX = pandas.read_csv('final_data.csv', names=colnamesX)
dataY = pandas.read_csv('final_data.csv', names=colnamesY)

# dividing X, y into train and test data 
X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, random_state = 0) 
  
# training a linear SVM classifier 
from sklearn.svm import SVC 
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 
svm_predictions = svm_model_linear.predict(X_test) 
  
# model accuracy for X_test   
accuracy = svm_model_linear.score(X_test, y_test) 
  
# creating a confusion matrix 
cm = confusion_matrix(y_test, svm_predictions) 






  
  
 
  
