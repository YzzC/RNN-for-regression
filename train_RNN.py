# import required packages
import q2_config

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import h5py

from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import load_model
from math import sqrt
from sklearn.metrics import mean_squared_error

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow

"""

the function that is used to generate training/testing data

def generate_data(df, ref_day):
    cols = list(df.columns)
    col_dict = {}
    for i, c in enumerate(cols):
        col_dict[c] = i
    print(col_dict)
    length = len(df)
    dataset = []
    for index in range(1, length-ref_day+1):
        d = {}
        
        target_date = df.iloc[index-1, col_dict['Date']]
        feature_date = list(df.iloc[index:index+ref_day, col_dict['Date']])
        
        target = df.iloc[index-1, col_dict[' Open']]
        
        feature_Volume = list(df.iloc[index:index+ref_day, col_dict[' Volume']])
        feature_Open = list(df.iloc[index:index+ref_day, col_dict[' Open']])
        feature_High = list(df.iloc[index:index+ref_day, col_dict[' High']])
        feature_Low = list(df.iloc[index:index+ref_day, col_dict[' Low']])
        
        features = feature_Volume + feature_Open + feature_High + feature_Low
        
        d['target_date'] = target_date
        d['feature_date'] = feature_date
        d['target'] = target
        for i, f in enumerate(features):
            d["feature"+str(i+1)] = f
        
        dataset.append(d)
    return dataset
"""

#data preprocessing
#training_data: feature1-3: volumn 4-6:open 7-9:High 10-12:Low
def data_preprocessing(X_train):
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train['feature1'] = min_max_scaler.fit_transform(X_train.feature1.values.reshape(-1,1))
    X_train['feature2'] = min_max_scaler.fit_transform(X_train.feature2.values.reshape(-1,1))
    X_train['feature3'] = min_max_scaler.fit_transform(X_train.feature3.values.reshape(-1,1))
    X_train['feature4'] = min_max_scaler.fit_transform(X_train.feature4.values.reshape(-1,1))
    X_train['feature5'] = min_max_scaler.fit_transform(X_train.feature5.values.reshape(-1,1))
    X_train['feature6'] = min_max_scaler.fit_transform(X_train.feature6.values.reshape(-1,1))
    X_train['feature7'] = min_max_scaler.fit_transform(X_train.feature7.values.reshape(-1,1))
    X_train['feature8'] = min_max_scaler.fit_transform(X_train.feature8.values.reshape(-1,1))
    X_train['feature9'] = min_max_scaler.fit_transform(X_train.feature9.values.reshape(-1,1))
    X_train['feature10'] = min_max_scaler.fit_transform(X_train.feature10.values.reshape(-1,1))
    X_train['feature11'] = min_max_scaler.fit_transform(X_train.feature11.values.reshape(-1,1))
    X_train['feature12'] = min_max_scaler.fit_transform(X_train.feature12.values.reshape(-1,1))
    X_train['target'] = min_max_scaler.fit_transform(X_train.target.values.reshape(-1,1))
       
    return X_train

def build_model(train_data, dropout_number, LSTM_units):
    model = Sequential()
    model.add(LSTM(LSTM_units,input_shape = (train_data.shape[1], train_data.shape[2]), return_sequences = True))
    model.add(Dropout(dropout_number))
    model.add(LSTM(LSTM_units,return_sequences = True))
    model.add(Dropout(dropout_number))
    model.add(LSTM(LSTM_units))
    model.add(Dropout(dropout_number))
    model.add(Dense(1,activation='relu'))
    
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

# process for optimizing the model

# def optimal_model(X_train, y_train, X_test, y_test, 
#                   dropout_number, LSTM_units, batch, epochs):
#     model = build_model(X_train, dropout_number, LSTM_units)
#     model.fit(X_train, y_train, batch_size = batch, epochs = epochs, validation_split = 0.1)
#     train_result, test_result = model_score1(model, X_train, y_train, X_test, y_test)
#     return train_result, test_result

# dropout_number_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
# LSTM_units_list = [32, 64, 128, 256]

# result = []
# for dr in dropout_number_list:
#     for unit in LSTM_units_list:
#         train_result, test_result = optimal_model(X_train, y_train, X_test, y_test,
#                                              dr, 128, 32, 100)
#         result.append({'dropout': dr, 'unit': unit, 'test_result': test_result})




if __name__ == "__main__": 
	# 1. load your training data
	TRAIN_DATA_PATH = q2_config.TRAIN_DATA_PATH
	train_data = pd.read_csv(TRAIN_DATA_PATH)

	normalization_train = data_preprocessing(train_data)

	#split the training feature and the training label
	Xtrain = pd.DataFrame(normalization_train, columns = ['feature1',
	                                                       'feature2',
	                                                       'feature3',
	                                                       'feature4',
	                                                       'feature5',
	                                                       'feature6',
	                                                       'feature7',
	                                                       'feature8',
	                                                       'feature9',
	                                                       'feature10',
	                                                       'feature11',
	                                                       'feature12'])
	y_train = pd.DataFrame(normalization_train, columns = ['target'])

	#LSTM input is 3D
	#reshape input to be 3D [samples, timesteps, features】
	Xtrain = Xtrain.values
	X_train = Xtrain.reshape((Xtrain.shape[0], 1, Xtrain.shape[1]))

	dropout_number = q2_config.dropout
	LSTM_units = q2_config.LSTM_units

	model = build_model(X_train, dropout_number, LSTM_units)


	# 2. Train your network
	# 		Make sure to print your training loss within training to show progress
	# 		Make sure you print the final training loss

	history = model.fit(X_train,y_train, batch_size = 32, epochs = 100, validation_split = 0.1)

	# 3. Save your model
	model.save(q2_config.model_save_path)

	plt.plot(history.history['loss'], label='train')
	plt.plot(history.history['val_loss'], label='val')
	plt.title('train_val', fontsize='12')
	plt.ylabel('loss', fontsize='10')
	plt.xlabel('epoch', fontsize='10')
	plt.legend()
	plt.show()


