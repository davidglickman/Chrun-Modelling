# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 11:48:38 2018

@author: David Glickman
"""

import numpy as np
import pandas as pd
import keras.layers
from keras.layers import Dense, Activation
import keras.models
import sklearn

chrunModellingDb = pd.read_csv('Churn_Modelling.csv')

""" split to data and results"""
chrunModellingRes = chrunModellingDb.iloc[:,13]
chrunModellingData = chrunModellingDb.iloc[:,3:13]

""" label encoding of nonnumeric data"""
le = sklearn.preprocessing.LabelEncoder()
chrunModellingData.iloc[:,2] = le.fit_transform(chrunModellingData.iloc[:,2])
chrunModellingData.iloc[:,1] = le.fit_transform(chrunModellingData.iloc[:,1])
chrunModellingData = sklearn.preprocessing.scale(chrunModellingData)

""" Buliding ANN """

chrunModellingANN = keras.models.Sequential()
chrunModellingANN.add(Dense(6, input_dim=10))    #input layer
chrunModellingANN.add(Activation('relu'))
chrunModellingANN.add(Dense(6, input_dim=6))    #hidden layer
chrunModellingANN.add(Activation('relu'))
chrunModellingANN.add(Dense(1, input_dim=6))    #output layer
chrunModellingANN.add(Activation('sigmoid'))

""" split the data to training and test segments"""
[x_train,x_test, y_train, y_test] = sklearn.model_selection.train_test_split(
        chrunModellingData,chrunModellingRes,test_size = 0.2, random_state=42)

chrunModellingANN.compile(loss='mean_squared_error', optimizer='sgd')

chrunModellingANN.fit(x_train, y_train, batch_size=None, epochs=1)

y_pred = chrunModellingANN.predict(x_test)

print(sklearn.metrics.accuracy_score(y_test, y_pred>0.5 , normalize=True, sample_weight=None))
