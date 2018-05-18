'''
Created on May 6, 2018

@author: nishant.sethi
'''
from scipy.io import loadmat
import numpy as np
import pandas as pd
from datetime import datetime, date, time

mat_file=loadmat('ex8_movies.mat')
mdata1=mat_file["R"]
mdata2=mat_file["Y"]
data_columns=[]
data1=pd.DataFrame(mdata1)
#data1.columns=["R"]
for i in range(1,944):
    data_columns.append("R-"+str(i))
data1.columns=data_columns
data1.info()
print(data1.head())
print("*"*40)
data2=pd.DataFrame(mdata2)
data_columns=[]
for i in range(1,944):
    data_columns.append("User-"+str(i))
data2.columns=data_columns
data2.info()
print(data2.head())
print("*"*40)
data2.to_csv("movie_rating.csv")
data1.to_csv("who_rated.csv")