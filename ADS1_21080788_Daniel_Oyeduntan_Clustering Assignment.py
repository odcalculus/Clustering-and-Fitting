#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 5 14:23:49 2023

@author: danieloyeduntan
"""
#Importing the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

import scipy.optimize as opt
import err_ranges as err

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

def model(x, a, b, c):
    """
    This is a function that takes in 5 arguments and returns a given function aX^3 + bX^2 + cX + d
    """
    x = x - 2000
    return a + b*x + c*x**2 

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


#Now the fitting part of the assignment, we read our file using a defined function
file = 'API_19_DS2_en_csv_v2_4773766.csv'
indicator = 'CO2 emissions (kg per PPP $ of GDP)'
cols = ['Country Name','1960', 
'1961', '1962', '1963', '1964', '1965', '1966', '1967', '1968', '1969', '1970', 
'1971', '1972', '1973', '1974', '1975', '1976', '1977', '1978', '1979', '1980', 
'1981', '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989', '1990', 
'1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', 
'2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', 
'2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', 
'2021']
df,df_T = data_read(file,indicator,cols)

#And then we do more preprocessing
df_T.columns = df_T.iloc[0]
df_T.drop('Country Name',inplace=True)
df_T['Year'] = df_T.index
df_fit = df_T[['Year','Spain']].dropna().apply(pd.to_numeric).values

#We now assign our x and y data to an array
x_axis = df_fit[:,0]
y_axis = df_fit[:,1]

#We then plot a scatterplot of the initial x and y axis
plt.figure(dpi=1000)
plt.title('Scatterplot of the initial x and y axis for fitting')
plt.xlabel('Year')
plt.ylabel('CO2 emissions (kg per PPP $ of GDP) for Spain')
plt.scatter(x_axis,y_axis)
plt.show()

#We then fit our curvefit model using the model function and our new x and y axis
popt, pcov = opt.curve_fit(model, x_axis, y_axis)

#We assign the output of the curve fit to variables and try to use err_ranges to get the lower and upper boundaries
a_opt, b_opt, c_opt = popt
year1 = 1990
year2 = 2030
x_line = np.arange(year1,year2+1,1)
y_line = model(x_line, a_opt, b_opt, c_opt)
sigma = np.sqrt(np.diag(pcov))
low, up = err.err_ranges(x_line, model, popt, sigma)


#Then we make a plot containing the scatterplot, curve fit line, and the curve boundaries
plt.figure(dpi=1000)
plt.scatter(x_axis, y_axis)
plt.plot(x_line, y_line,c='black')
plt.fill_between(x_line, low, up, alpha=0.2, color='blue')
plt.title('Plot of the x and y axis fitted with future predictions')
plt.xlabel('Year')
plt.ylabel('CO2 emissions (kg per PPP $ of GDP) for Spain')
plt.show()

#We we create a dataframe that will show the predicted values for future years
future_predictions = pd.DataFrame({'Year':x_line,'Predicted':y_line})
future_predictions = future_predictions.iloc[30:]
print(future_predictions)