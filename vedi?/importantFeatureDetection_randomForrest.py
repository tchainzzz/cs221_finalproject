# for data
import csv

# for data management
import pandas as pd

# for mathematical operations
import numpy as np
import math

# for data preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle

# our model
#from sklearn import svm
#from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
#from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# for data analysis
from sklearn.metrics import confusion_matrix
from sklearn import metrics

# for visualization
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import seaborn as sns

# for command line interface
import argparse
import sys

PATH_NAME = './csv/final_data.csv'

if __name__ == '__main__':
	# command line setup
    psr = argparse.ArgumentParser()
    psr.add_argument("-q", "--quiet", help="No plot", action="store_true")
    psr.add_argument("-p", "--path-to-dataframe", help="Path to dataframe", type=str, default=PATH_NAME)
    args = psr.parse_args()


    print("Loading data...")

    #########################################
    #
    #  Feature Selection:
    #  We set colnamesX to the selected features we want. The first two will be graphed.
    #
    ##########################################
    colnamesX = ['APG_y', 'RPG_y','PPG_y', 'FG%_y',  'STPG_y', 'BLKPG_y', 'GP_y', 'MPG_y', '2P%_y', '3P%_y', 'PTS_x', 'DRPG_x', 'TOPG_x', 'PF_x', 'RPG_x']
    #colnamesX = ['PTS_x', 'DRPG_x', 'TOPG_x', 'PF_x', 'RPG_x']
    colnamesY = ['Class'] # class label

    # read in data
    bigDF = shuffle(pd.read_csv(args.path_to_dataframe)) # shuffle data as we read it in
    dataX = bigDF[colnamesX] # feature data
    #dataX = dataX.fillna(dataX.mean())
    dataY = bigDF[colnamesY] # labels
    print("Feature vector table shape:", dataX.shape)
    print("Label table shape:", dataY.shape)

    dataX.to_csv("./csv/data_x.csv") 
    dataY.to_csv("./csv/data_y.csv")

    ################################
    #
    # Option one: transform data to a normal distribution. Uncomment these two lines.
    #
    ################################
    #scalerX = StandardScaler()
    #newDataX = scalerX.fit_transform(X=dataX.astype('float64'))

    ################################
    #
    # Option two: Don't transform the data. This is what this line is.
    #
    ################################
    newDataX = dataX.values


    # dividing X, y into train and test data 
    X_train, X_test, y_train, y_test = train_test_split(newDataX, dataY.values.ravel(), test_size=0.25) 


    print("Training model...")
    #Create a Gaussian Classifier
    clf=RandomForestClassifier(n_estimators=100)

    #Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train,y_train)

    feature_imp = pd.Series(clf.feature_importances_,index=colnamesX).sort_values(ascending=False)
    feature_imp
    print(feature_imp)

    # Creating a bar plot
    sns.barplot(x=feature_imp, y=feature_imp.index)
    # Add labels to your graph
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Important Features")
    plt.legend()
    plt.show()




