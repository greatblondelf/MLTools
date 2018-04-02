#!/usr/bin/env python2
from __future__ import division
#import Constants
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option('display.max_colwidth', -1) # show all the columns/data.
pd.set_option("display.max_columns",201)

def displayDataScatterplots(df,keep_columns,colorCol):
    #display only the chosen columns as a multivariate scatterplot
    df_display = Dataset.Dataset()
    df_display.set_data(df.data())
    colors=['blue','red']
    colorInd = df_display.data()[colorCol].tolist()
    colorInd = [int(x*1.0) for x in colorInd]
    colorMap = [ colors[i] for i in colorInd]
    df_display.drop_columns(get_columns_to_drop(df,keep_columns))
    #print(df_display.data()[colorCol].tolist())
    myplot = scatter_matrix(df_display.data(), alpha=0.2, figsize=(6, 6), diagonal='kde', c=colorMap,s=100)
    plt.rcParams['axes.labelsize'] = 10
    plt.show()

#############################################################


#take all the data and put it into a pandas dataframe
input_data = pd.read_csv('data/input_data.csv')



#print(input_data.head())

unpaid_ads = input_data[input_data['Lifetime Post Paid Reach'] == 0]
paid_ads = input_data[input_data['Lifetime Post Paid Reach'] != 0]

print("unpaid ads:")
print(unpaid_ads.head())
print("paid ads:")
print(paid_ads.head())

\
