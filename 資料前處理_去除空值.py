import pandas as pd
import numpy as np

data = pd.read_csv('C:\\Users\\wxy\\PycharmProjects\\project108\\DM\\adult.csv')
print(data.isin(['?']).sum())
print(np.shape(data))
print("=========================================")

data_del = data.replace(to_replace ='?', value =np.nan)
data_del.dropna(axis=0, how='any', inplace=True)
print(data_del.isin([np.nan]).sum())
print(np.shape(data_del))
data_del.to_csv("C:\\Users\\wxy\\PycharmProjects\\project108\\DM\\adult_del.csv", index=False)
