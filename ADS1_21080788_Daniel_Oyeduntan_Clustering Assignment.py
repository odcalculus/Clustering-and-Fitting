#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 5 14:23:49 2023

@author: danieloyeduntan
"""
#Importing the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

#Definition of functions to be used
def data_read(filename,indicators,cols):
    """
    This is a function that reads csv file and returns an output of two data frames. 
    It takes 4 arguments:
    1. The filename
    2. The indicator
    3. Columns to drop
    """
    data = pd.read_csv(filename,skiprows=4)
    data = data[data['Indicator Name'] == indicators]
    data = data[cols]
    data.reset_index(drop=True,inplace=True)
    return data,data.transpose()

def sieve_rows(df1,col1,df2,col2):
    """
    This is a function that filters rows of one dataframe based on whether or not it is found in the second dataframe.
    It takes 4 arguments: 
    1. The first dataframe
    2. The column to be filtered based on in the first dataframe
    3. The second dataframe
    4. The column to be checked in the second dataframe.
    """
    data = df1[df1[col1].isin(df2[col2])]
    data.reset_index(drop=True,inplace=True)
    return data

def scale_data(df):
    """
    This is a function that normalises dataframe passed into it.
    It takes just one argument - the dateframe to be scaled.
    """
    headers = df.columns
    scalar = MinMaxScaler()
    scalar.fit(df)
    scaled_features = scalar.transform(df)
    data = pd.DataFrame(scaled_features,columns=headers)
    return data

#Reading of the file using a predefined function
file = 'API_19_DS2_en_csv_v2_4773766.csv'
indicator = 'Urban population growth (annual %)'
cols = ['Country Name', '1980', '2021']
df,df_T = data_read(file,indicator,cols)

#We want to filter just countries from the dataframe, hence we would read a txt file of all countries and pass it through a predefined function
country = pd.read_csv('text.txt',header=None)
country.columns=['Name']
df = sieve_rows(df,'Country Name',country,'Name')

#We then plot a scatter plot to show our data
plt.figure(dpi=1000)
plt.title('Scatter plot showing the annaul Urban population growth - 1980 & 2021',fontsize=9)
plt.scatter(df['1980'],df['2021'])
plt.xlabel('1980')
plt.ylabel('2021')
plt.legend
plt.show()

#Next we standardize/normalize our data
df_std = df[['1980', '2021']]
df_std = scale_data(df_std)

#To help determine the number of clusters to use
sse = []
k_range = range(1,11)
for k in k_range:
    km = KMeans(n_clusters=k)
    km.fit(df_std)
    sse.append(km.inertia_)
    
plt.figure(dpi=1000)
plt.plot(k_range,sse)
plt.title('Elbow method to decide number of clusters')
plt.show()

#Fitting our data on 5 clusters
km = KMeans(n_clusters=5)
predicted = km.fit_predict(df_std)

df['Clusters'] = predicted
print(df)

#Scatter plot showing the data in different clusters
df0 = df[df['Clusters']==0]
df1 = df[df['Clusters']==1]
df2 = df[df['Clusters']==2]
df3 = df[df['Clusters']==3]
df4 = df[df['Clusters']==4]

df_list = [df0, df1, df2, df3, df4]
df_color = ['green','red','blue','yellow','purple']

plt.figure(dpi=1000)
plt.title('Scatter plot showing all different clusters and position of centroid',fontsize=9)
for i in range(len(df_list)):
    plt.scatter(df_list[i]['1980'],df_list[i]['2021'],c=df_color[i])
#plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],c='black',marker='*',label='centroid')
plt.xlabel('1980')
plt.ylabel('2021')
plt.legend
plt.show()