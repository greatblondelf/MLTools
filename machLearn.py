#!/usr/bin/env python2
from __future__ import division
#import Constants
#import Dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, precision_score, recall_score

pd.set_option('display.max_colwidth', -1) # show all the columns/data.
pd.set_option("display.max_columns",201)

success_thresh = 6000 #threshold to be considered successful
use_columns = ['Type','Persona Target','Topic', 'Image Type', 'Image Background', 'Tone','Sentiment','Comment Length', 'Question Asked?', 'Lifetime Post Total Reach']
#############################################################

#take all the data and put it into a pandas dataframe
input_data = pd.read_csv('data/input_data.csv')
#input_data['quest'] = (input_data['Question Asked?'] == 'Yes') * 1

#throw out outliers (manual inspection) and unneded columns
input_data = input_data[input_data['Comment Length'] < 2000]
input_data = input_data[use_columns]

input_data = pd.get_dummies(input_data, columns=['Type','Persona Target','Topic', 'Image Type', 'Image Background', 'Tone','Sentiment', 'Question Asked?'])
print(input_data.head())

#input_data = input_data.drop(['adxprob_adn','adxprob_dfp','dfp_company','adnet_network','ad_id','ad_frame_tree','result_path'], axis=1)

#Split the data sets into training and test
n_samples = len(input_data)
msk = np.random.rand(n_samples) < 0.7
trainData = input_data[msk]
testData = input_data[~msk]
print("length of training data set is %s samples" % len(trainData))
print("length of test data set is %s samples" % len(testData))
trainTarget = (trainData['Lifetime Post Total Reach'] > success_thresh) * 1
testTarget = (testData['Lifetime Post Total Reach'] > success_thresh) * 1

#make sure we aren't using the target as a thing to train on!
trainData = trainData.drop(['Lifetime Post Total Reach'], axis=1)
testData = testData.drop(['Lifetime Post Total Reach'], axis=1)

model = RandomForestClassifier(n_estimators=20)
model.fit(trainData, trainTarget)
print(model)

expected = testTarget
predicted = model.predict(testData)
print('Confusion Matrix:')
q = confusion_matrix(expected, predicted)
print q
#precision :  What are the odds that a guessed "yes" really is yes
#recall: How many actual "yes" results did we catch?
print('precision is %s , recall is %s' % (precision_score(expected, predicted),recall_score(expected, predicted)))

#keep the model parameters
#joblib.dump(model, './models/RF_tunedmodel.pkl')
#Later you can load back the pickled model (possibly in another Python process) with:
#clf = joblib.load('filename.pkl')
