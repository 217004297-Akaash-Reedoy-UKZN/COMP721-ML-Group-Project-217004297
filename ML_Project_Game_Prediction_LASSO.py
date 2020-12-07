# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 14:37:11 2020

@author: Akaash Reedoy
"""

import pandas as pd
import csv

data = pd.read_csv("team_season.csv")
y_names = data.iloc[:, [0,1]] #team name, year

data = data[data['year']>=1980]
X_train = data.iloc[:len(data)-30, [3, 4, 5, 7, 10, 13, 15, 18, 19, 20, 28, 29, 30, 34, 35]]
X_test = data.iloc[len(data)-30:, [3, 4, 5, 7, 10, 13, 15, 18, 19, 20, 28, 29, 30, 34, 35]]
X_train_win_ratio = X_train.iloc[:, [13, 14]]
X_test_win_ratio = X_test.iloc[:, [13, 14]]
X_train_win_ratio = X_train_win_ratio.values
X_test_win_ratio = X_test_win_ratio.values

winLossRatio = []
for i in range(0, len(X_train_win_ratio)):
    winLossRatio.append(X_train_win_ratio[i][0]/(X_train_win_ratio[i][0]+X_train_win_ratio[i][1]))

X_train = X_train.drop(['won', 'lost'], 1)

y_train = pd.DataFrame(winLossRatio)
y_train.columns = ['Win Ratio']

winLossRatio = []
for i in range(0, len(X_test_win_ratio)):
    winLossRatio.append(X_test_win_ratio[i][0]/(X_test_win_ratio[i][0]+X_test_win_ratio[i][1]))
print("Test target:\n",winLossRatio)

X_test = X_test.drop(['won', 'lost'], 1)

y_test = pd.DataFrame(winLossRatio)
y_test.columns = ['Win Ratio']

with open('team_season.csv', newline='') as f:
  reader = csv.reader(f)
  header = next(reader)

print(X_train.head(100))

#LASSO
from sklearn.linear_model import Lasso

lasso_model = Lasso().fit(X_train, y_train)
r_squared_Lasso = lasso_model.score(X_test, y_test)

print('Lasso coefficient of determination:', r_squared_Lasso)

y_pred = lasso_model.predict(X_test)
print(y_pred)

X_test['team'] = data['team']
X_test['Actual'] = y_test.values
X_test['Prediction'] = y_pred

print(X_test.head(50))

import sklearn.metrics as sm

print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_pred), 2)) 
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_pred), 2)) 
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_pred), 2)) 
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_pred), 2)) 
print("R2 score =", round(sm.r2_score(y_test, y_pred), 2))

team1name = input("Please enter team 1:")
team2name = input("Please enter team 2:")

team1 = X_test[X_test['team']==team1name]
team2 = X_test[X_test['team']==team2name]

team1WinRatio = team1['Prediction'].values
team2WinRatio = team2['Prediction'].values

if team1WinRatio>team2WinRatio:
    print(team1name, "beats", team2name)
elif team2WinRatio>team1WinRatio:
    print(team2name, "beats", team1name)
else:
    print("Draw")


