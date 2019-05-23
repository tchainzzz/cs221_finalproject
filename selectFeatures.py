import csv
from sklearn import svm
import numpy as np
import pandas as pd
from sklearn import datasets 
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_decision_regions
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
from sklearn.svm import SVC 


if __name__ == '__main__':
    colnamesX = ['PTS_x', 'DRPG_x', 'TOPG_x', 'PF_x', 'RPG_x']
    colnamesY = ['Class']
    PATH_NAME = './csv/final_data.csv'
    dataX = pd.read_csv(PATH_NAME)[colnamesX]
    dataY = pd.read_csv(PATH_NAME)[colnamesY]

    dataX.to_csv("./csv/data_x.csv")
    dataY.to_csv("./csv/data_y.csv")

    # dividing X, y into train and test data 
    X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, random_state = 0) 
    
    # training a linear SVM classifier 
    svm_model_linear = OneVsOneClassifier(SVC(kernel = 'linear', C = 1)).fit(X_train, y_train)
    svm_predictions = svm_model_linear.predict(X_test) 
    
    # model accuracy for X_test   
    accuracy = svm_model_linear.score(X_test, y_test) 
    
    # creating a confusion matrix 
    cm = confusion_matrix(y_test, svm_predictions) 

    widths = {2 + ix :abs(dataX[col].max() - dataX[col].min()) / float(2) for ix, col in enumerate(colnamesX[2:])}
    values = {2 + ix :(dataX[col].max() - dataX[col].min()) / float(2) for ix, col in enumerate(colnamesX[2:])}
    _, ax = plt.subplots()

    ax.set_xlabel('Points per game')
    ax.set_ylabel('Defensive rebounds per game')
    plot_decision_regions(X=np.array(dataX.astype(np.integer)), 
        y=np.array(dataY['Class'].astype(np.integer)), 
        filler_feature_values=values,
        filler_feature_ranges=widths,
        clf=svm_model_linear, legend=2)
    plt.show()







  
  
 
  
