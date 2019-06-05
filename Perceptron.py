# for data
import csv

# for data management
import csv
import pandas as pd

# for mathematical operations
import numpy as np
import math

# Load required libraries
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import fancyimpute as fi

# for data analysis
from sklearn.metrics import confusion_matrix

# for visualization
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import seaborn as sn

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
    #colnamesX = ['APG_y', 'RPG_y','PPG_y', 'FG%_y',  'STPG_y', 'BLKPG_y', 'GP_y', 'MPG_y', '2P%_y', '3P%_y']
    colnamesX = ['PTS_x', 'DRPG_x', 'TOPG_x', 'PF_x', 'RPG_x']
    #colnamesX = ['APG_y', 'RPG_y','PPG_y', 'FG%_y',  'STPG_y', 'BLKPG_y', 'GP_y', 'MPG_y', '2P%_y', '3P%_y']
    #colnamesX = ['PTS_x','PF_x','TOPG_x']
    colnamesY = ['Class'] # class label
    allCols = colnamesX + colnamesY
    # read in data
    bigDF = shuffle(pd.read_csv(args.path_to_dataframe)) # shuffle data as we read it in


    # dataX.to_csv("./csv/data_x.csv") 
    # dataY.to_csv("./csv/data_y.csv")

    focusDF = bigDF[allCols].dropna(thresh=2)
    dataY = focusDF[colnamesY] # labels
    dataX = focusDF[colnamesX]
    dataX = fi.KNN(k=3).fit_transform(dataX)
    print("Feature vector table shape:", dataX.shape)
    print("Label table shape:", dataY.shape)


    # dividing X, y into train and test data 
    X_train, X_test, y_train, y_test = train_test_split(dataX, dataY.values.ravel(), test_size=0.25) 
    scalerX = MinMaxScaler()
    newDataX = scalerX.fit(X_train)
    # newDataX = dataX

    # Apply the scaler to the X training data
    X_train_std = scalerX.transform(X_train)

    # # Apply the SAME scaler to the X test data
    X_test_std = scalerX.transform(X_test)

    # Create a perceptron object with the parameters: 40 iterations (epochs) over the data, and a learning rate of 0.1
    ppn = Perceptron(max_iter=100000, eta0=0.05, tol=1e-4, penalty='elasticnet')

    # Train the perceptron
    ppn.fit(X_train, y_train)

    # Apply the trained perceptron on the X data to make predicts for the y test data
    y_pred = ppn.predict(X_test)

    print(y_pred)

    print(y_test)

    # View the accuracy of the model, which is: 1 - (observations predicted wrong / total observations)
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

    # creating a confusion matrix. We should have a nice diagonal line.
    cm = confusion_matrix(y_test, y_pred) 
    print("Confusion matrix:\n", cm)

    # These are settings for drawing. Do whatever with these.
    
    widths = {2 + ix : 10 for ix, col in enumerate(colnamesX[2:])}
    values = {2 + ix : np.median(dataX[ix]) for ix, col in enumerate(colnamesX[2:])}
    figs, ax = plt.subplots(2, 1, figsize=(8, 10))
    print("Features shown in graph:", colnamesX[:2])
    print("Features flattened in graph:",colnamesX[2:])
    #figs, ax = plt.subplots(len(colnamesX) + 1, len(colnamesX), figsize=(3, 4))


    ax[0].set_xlabel(colnamesX[0])
    ax[0].set_ylabel(colnamesX[1])
    print("Plotting main plot...")
    figs = plot_decision_regions(X=X_test, 
        y=y_test, 
        filler_feature_values=values,
        filler_feature_ranges=widths,
        clf=ppn, legend=1, ax=ax[0])
    print("Plotting confusion matrix...")
    sn.heatmap(cm, annot=True,annot_kws={"size": 10})

    # figs, axes = plt.subplots(len(colnamesX), len(colnamesX), sharex='col', sharey='row',
    #                             figsize=(10, 10))
    # for i in range(len(colnamesX)):
    #     for j in range(len(colnamesX)):
    #         if i == j: continue
    #         arraySlice = [x for (ii, x) in enumerate(colnamesX) if (ii != i and ii != j)]
    #         widths = {2 + ix : np.max(newDataX[ix]) for ix, col in enumerate(arraySlice)}
    #         arraySlice = enumerate([x for (ii, x) in enumerate(colnamesX) if (ii != i and ii != j)])
    #         values = {2 + ix : np.median(newDataX[ix]) for ix, col in enumerate(arraySlice)}
    #         figs = plot_decision_regions(X=newDataX, 
    #             y=np.array(dataY['Class'].astype(np.integer)), 
    #             filler_feature_values=values,
    #             filler_feature_ranges=widths,
    #             clf=ppn, ax=axes[i, j], legend=0)
    # # print("Plotting confusion matrix...")
    # # sn.heatmap(cm, annot=True,annot_kws={"size": 10})
    # print("Features shown in graph:", colnamesX[:2])
    # print("Features flattened in graph:",colnamesX[2:])

    # for ax, col in zip(axes[0], colnamesX): ax.set_title(col, size='medium')
    # for ax, row in zip(axes[:, 0], colnamesX): ax.set_ylabel(col, size='medium')

    print("Plotting main plot...")

    plt.show()






