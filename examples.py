#!/usr/bin/env python2
from __future__ import division
#import Constants
#import Dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing

pd.set_option('display.max_colwidth', -1) # show all the columns/data.
pd.set_option("display.max_columns",201)

def displayDataScatterplots(df,keep_columns,colorCol):
    #display only the chosen columns as a multivariate scatterplot
    df_display = df[keep_columns]
    colors=['blue','red']
    colorInd = df_display[colorCol].tolist()
    colorInd = [int(x*1.0) for x in colorInd]
    colorMap = [ colors[i] for i in colorInd]
    myplot = pd.plotting.scatter_matrix(df_display, alpha=0.2, figsize=(6, 6), diagonal='kde', c=colorMap,s=100)
    plt.rcParams['axes.labelsize'] = 10
    plt.show()
#############################################################

#take all the data and put it into a pandas dataframe
input_data = pd.read_csv('data/input_data.csv')
input_data['quest'] = (input_data['Question Asked?'] == 'Yes') * 1
input_data['paid'] = (input_data['Lifetime Post Paid Reach'] != 0) * 1

#throw out outliers (manual inspection)
input_data = input_data[input_data['Comment Length'] < 2000]

#unpaid_ads = input_data[input_data['Lifetime Post Paid Reach'] == 0]
#paid_ads = input_data[input_data['Lifetime Post Paid Reach'] != 0]
#print("unpaid ads:")
#print(unpaid_ads.head())
#print("paid ads:")
#print(paid_ads.head())

le_topic = preprocessing.LabelEncoder()
le_topic.fit(input_data['Topic'])
input_data['Topic_num'] = le_topic.transform(input_data['Topic'])
print('Topic Numbers correspond to:')
print(list(le_topic.inverse_transform(list(range(  input_data['Topic_num'].max()+1  )))))

le_Persona_Target = preprocessing.LabelEncoder()
le_Persona_Target.fit(input_data['Persona Target'])
input_data['Persona_Target_num'] = le_Persona_Target.transform(input_data['Persona Target'])
print('Persona Target Numbers correspond to:')
print(list(le_Persona_Target.inverse_transform(list(range(  input_data['Persona_Target_num'].max()+1  )))))

plot_colnames = ['paid','quest','Comment Length','Lifetime Post Total Reach', 'Topic_num','Persona_Target_num']
color_col_name = 'paid'

displayDataScatterplots(input_data,plot_colnames,color_col_name)
