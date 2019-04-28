from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

nRowsRead = 1000 # specify 'None' if want to read whole file
# historical_projections.csv has 1090 rows in reality, but we are only loading/previewing the first 1000 rows
df1 = pd.read_csv('./historical_projections.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'historical_projections.csv'
nRow, nCol = df1.shape

fig = plt.figure()
plt.title('Correlation matrix for 2001-2015 NBA Draft picks')
ax1 = fig.add_subplot(111)
cax = ax1.imshow(df1.corr(), interpolation="nearest")
labels = list(df1)
ax1.set_xticklabels(labels)
ax1.set_yticklabels(labels)
fig.colorbar(cax)

plt.show()


