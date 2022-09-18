# 資料前處理(正規化)
import pandas as pd
import numpy as np

data = pd.read_csv('C:\\Users\\wxy\\PycharmProjects\\project108\\DM\\education_group.csv')
print(np.shape(data))

WorkClass = {
	'Self-emp-inc': 1.0, 'Self-emp-not-inc': 2.0, 'Private': 3.0,
	'Federal-gov': 4.0, 'State-gov': 5.0,'Local-gov': 6.0,
	'Never-worked': 7.0, 'Without-pay': 8.0}

MaritalStatus = {
	'Married-civ-spouse': 1.0, 'Married-AF-spouse': 2.0, 'Married-spouse-absent': 3.0,
	'Divorced': 4.0, 'Separated': 5.0, 'Widowed': 6.0, 'Never-married': 7.0}

Occupation = {
	'Handlers-cleaners': 1.0, 'Craft-repair': 2.0, 'Other-service': 3.0,
	'Sales': 4.0, 'Machine-op-inspct': 5.0,'Exec-managerial': 6.0, 'Prof-specialty': 7.0,
	'Tech-support': 8.0, 'Adm-clerical': 9.0, 'Farming-fishing': 10.0, 'Transport-moving': 11.0,
	'Priv-house-serv': 12.0, 'Protective-serv': 13.0, 'Armed-Forces': 14.0}

Sex = {'Female': 1.0, 'Male': 2.0}

Income = {'>50K': 1.0, '<=50K': 0.0}

data['workclass'] = data['workclass'].map(WorkClass)
data['marital.status'] = data['marital.status'].map(MaritalStatus)
data['occupation'] = data['occupation'].map(Occupation)
data['sex'] = data['sex'].map(Sex)
data['income'] = data['income'].map(Income)

data = data.drop('age', axis=1)
data = data.drop('education', axis=1)
data = data.drop('hours.per.week', axis=1)
data.to_csv("C:\\Users\\wxy\\PycharmProjects\\project108\\DM\\adult_fin.csv", index=False)