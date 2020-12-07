# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 21:14:45 2020

@author: Akaash Reedoy
"""

import pandas as pd
from sklearn import preprocessing
import csv

#LOADING DATA Start
data = pd.read_csv("player_regular_season_career.csv")
X_train = data.iloc[:, 4:]
org = X_train
y_names = data.iloc[:, [0,1,2]]

with open('player_regular_season_career.csv', newline='') as f:
  reader = csv.reader(f)
  header = next(reader)

x = X_train.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

X_train = pd.DataFrame(x_scaled)
X_train.columns =header[4:]
#LOADING DATA End 


###############################################################################


#DBSCAN Start
from sklearn.cluster import DBSCAN
outlier_detection = DBSCAN(eps = 0.60, metric="euclidean", min_samples = 50, n_jobs = 1)
clusters = outlier_detection.fit(X_train)

outlier_df = pd.DataFrame(X_train)

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_axes([.1,.1,1,1])
colours = clusters.labels_

from matplotlib import cm
cmap = cm.get_cmap('Set1')

ax.scatter(X_train.iloc[:,2].values, X_train.iloc[:,1].values, c=colours, s=20, cmap=cmap, edgecolor="black")

ax.set_xlabel("Points")
ax.set_ylabel("Minutes")

plt.title("DBSCAN Outlier Detection")
plt.show()

outliers = outlier_df[clusters.labels_==-1]
outliers = outliers[:].index.to_numpy()

y_names = y_names.values

print("DBSCAN OUTLIERS:")
for i in range(0, len(outliers)):
    print(outliers[i], "-> ", y_names[outliers[i]])
print("Number of outliers:", len(outliers), "\n********************END********************")    
#DBSCAN End
    
    
###############################################################################
    
    
#LOF Anomaly detection Start
from sklearn.neighbors import LocalOutlierFactor
clf = LocalOutlierFactor(n_neighbors=30, contamination=.006)
y_pred = clf.fit_predict(X_train)
LOF_Scores = clf.negative_outlier_factor_
LOF_pred=pd.Series(y_pred).replace([-1,1],[1,0])
LOF_anomalies=X_train[LOF_pred==1]
plt.scatter(X_train.iloc[:,2],X_train.iloc[:,1],c='grey',s=20,edgecolor='black')
plt.scatter(LOF_anomalies.iloc[: ,2], LOF_anomalies.iloc[: ,1], c='red', edgecolor='black')
plt.title('LOF Outlier Detection')
plt.ylabel('Minutes')
plt.xlabel('Points')
plt.show()

indexes = LOF_anomalies.index
indexes = list(indexes)

print("\nLOF OUTLIERS:")
for i in range(0, len(indexes)):
    print(indexes[i], "->", y_names[indexes[i]])
print("Number of outliers:", len(indexes), "\n********************END********************") 
#LOF Anomaly detection End

###############################################################################

#ONE-CLASS SVM Start

from sklearn import svm
clf_svm = svm.OneClassSVM(nu=.007 , kernel='rbf', gamma=.001)
clf_svm.fit(X_train)
y_pred = clf_svm.predict(X_train)

plt.scatter(X_train.iloc[:,2].values,X_train.iloc[:,1].values,c=y_pred, cmap=cmap,s=20,edgecolor='black')
plt.title("ONE-CLASS SVM Outlier Detection")
plt.show()

outlier_list = []
for i in range(0, len(y_pred)):
    if y_pred[i]==-1:
        outlier_list.append(i)
        
print("ONE-CLASS SVM OUTLIERS:")
for i in range(0, len(outlier_list)):
    print(outlier_list[i], "-> ", y_names[outlier_list[i]])
print("Number of outliers:", len(outlier_list), "\n********************END********************")   

#ONE-CLASS SVM End


###############################################################################

