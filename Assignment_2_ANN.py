# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 15:24:59 2022

@author: Frank Wan
"""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Import Libraries Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn import preprocessing
import matplotlib.pyplot as plt

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Parameters Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""
0 = fixed_acidity
1 = volatile_acidity
2 = citric_acid
3 = residual_sugar
4 = chlorides
5 = free_sulfur_dioxide
6 = total_sulfur_dioxide
7 = density
8 = pH
9 = sulphates
10= alcohol
11 = quality
"""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Load Data Section & Pretreat Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
wine = pd.read_csv("C:/Users/Frank Wan/Desktop/MSBA Files/Semester 2/AI/assignment 2/winequality-white.csv",delimiter=";")
#print('wine.shape=',wine.shape) #view data shape
wine = wine.values 
iv = wine[:,0:11] #this gives us the independent variables
dv = wine[:,11] #this gives us the dependent variable - quality
#print('iv.shape:',iv.shape)
#print('dv.shape:',dv.shape)
iv_MinMax = preprocessing.MinMaxScaler()
iv = iv_MinMax.fit_transform(iv)
iv.mean(axis=0)
#print('iv_MinMax.scale_=',iv_MinMax.scale_)

iv_train,iv_test,dv_train, dv_test = train_test_split(iv, dv, test_size=0.2)
#print('iv_train.shape:',iv_train.shape)
#print('iv_test.shape:',iv_test.shape)
#print('dv_train.shape:',dv_train.shape)
#print('dv_test.shape:',dv_test.shape)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Define Model Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
start_time = datetime.datetime.now()
#create model
MAE_total={}
for n_nodes_h1 in [49, 50, 51]:
    for n_nodes_h2 in [77, 78, 79]:
        model=Sequential()
        model.add(Dense(n_nodes_h1,input_dim=11, activation='relu')) #first layer (13), input is 13 columns, 
        model.add(Dense(n_nodes_h2, activation='relu'))
        model.add(Dense(1)) #output is 1
        # in this section, we can change the hidden layer's number, and we can also add another hidden layer (helps us to find more complex features to play around)
        
        
        #compile model
        model.compile(loss='mean_squared_error',optimizer='adam',metrics=['mean_absolute_error'])
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        Train Model Section
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        history=model.fit(iv_train, dv_train, epochs=10, batch_size=5, verbose=0, validation_data=(iv_test,dv_test)) #saves the value of loss function, epochs is the number of times we go through training data
        
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        Show output Section
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        print('n_nodes_h1=',n_nodes_h1)
        print('n_nodes_h2=',n_nodes_h2)
        score=model.evaluate(iv_test,dv_test,verbose=0)
        print('Test Loss=',score[0])
        print('Test MAE=',score[1])
        
        print("")
        key=(n_nodes_h1,n_nodes_h2)
        MAE_total[key]=score[1]
        
        #visualization training history
        plt.figure()
        plt.plot(history.history['mean_absolute_error'],label='Training accuracy')
        plt.plot(history.history['val_mean_absolute_error'],label='Test accuracy')
        plt.title('Training / Test MAE values')
        plt.xlabel('Epach')
        plt.ylabel('MAE')
        plt.legend(loc="upper right")
        plt.show()
        
        
        #visualization training loss history
        plt.figure()
        plt.plot(history.history['loss'],label='Training loss')
        plt.plot(history.history['val_loss'],label='Test loss')
        plt.title('Training / Test loss values')
        plt.xlabel('Epach')
        plt.ylabel('loss')
        plt.legend(loc="upper right")
        plt.show()
print('Best combination of hidden layers: ',min(MAE_total.items(),key=lambda x:x[1]))
stop_time = datetime.datetime.now()
print ("Time required for training:",stop_time - start_time)






























