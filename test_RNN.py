# import required packages
import q2_config

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import h5py

from sklearn import preprocessing
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import load_model
from math import sqrt
from sklearn.metrics import mean_squared_error

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow

def data_preprocessing(X_test):
    min_max_scaler = preprocessing.MinMaxScaler()
    X_test['feature1'] = min_max_scaler.fit_transform(X_test.feature1.values.reshape(-1,1))
    X_test['feature2'] = min_max_scaler.fit_transform(X_test.feature2.values.reshape(-1,1))
    X_test['feature3'] = min_max_scaler.fit_transform(X_test.feature3.values.reshape(-1,1))
    X_test['feature4'] = min_max_scaler.fit_transform(X_test.feature4.values.reshape(-1,1))
    X_test['feature5'] = min_max_scaler.fit_transform(X_test.feature5.values.reshape(-1,1))
    X_test['feature6'] = min_max_scaler.fit_transform(X_test.feature6.values.reshape(-1,1))
    X_test['feature7'] = min_max_scaler.fit_transform(X_test.feature7.values.reshape(-1,1))
    X_test['feature8'] = min_max_scaler.fit_transform(X_test.feature8.values.reshape(-1,1))
    X_test['feature9'] = min_max_scaler.fit_transform(X_test.feature9.values.reshape(-1,1))
    X_test['feature10'] = min_max_scaler.fit_transform(X_test.feature10.values.reshape(-1,1))
    X_test['feature11'] = min_max_scaler.fit_transform(X_test.feature11.values.reshape(-1,1))
    X_test['feature12'] = min_max_scaler.fit_transform(X_test.feature12.values.reshape(-1,1))
    X_test['target'] = min_max_scaler.fit_transform(X_test.target.values.reshape(-1,1))
       
    return X_test

def denormalization(df, pred):
    normalize_data = pred.reshape(-1,1)
    min_max_scaler = preprocessing.MinMaxScaler()
    x = df['target'].values.reshape(-1,1)
    obj = min_max_scaler.fit_transform(x)
    ans = min_max_scaler.inverse_transform(normalize_data)
    return ans

if __name__ == "__main__":
	# 1. Load your saved model
	model = load_model(q2_config.model_load_path)

	# 2. Load your testing data
	TEST_DATA_PATH = q2_config.TEST_DATA_PATH
	test_data = pd.read_csv(TEST_DATA_PATH)
	origin_test = pd.read_csv(TEST_DATA_PATH)

	normalization_test = data_preprocessing(test_data)

	Xtest = pd.DataFrame(normalization_test, columns = ['feature1',
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
	y_test = pd.DataFrame(normalization_test, columns = ['target'])

	Xtest = Xtest.values
	X_test = Xtest.reshape((Xtest.shape[0], 1, Xtest.shape[1]))

	# 3. Run prediction on the test data and output required plot and loss
	model.evaluate(X_test, y_test)

	pred = model.predict(X_test)
	prediction = denormalization(origin_test, pred)
	prediction = pd.DataFrame(prediction)
	df = pd.concat( [origin_test, prediction], axis=1 )
	df['Date'] = pd.to_datetime(df['target_date'])
	df.index = df['Date']
	df = df.iloc[:, :-1]
	df = df.sort_index()

	y_pred = df[0]
	y_act = df['target']


	plt.plot(y_pred, color='red', label='Prediction')
	plt.plot(y_act,color='yellow', label='Actual')
	plt.legend(loc='best')
	plt.title('Prediction vs Actual')
	plt.xlabel('Days')
	plt.ylabel('Open')

	plt.show()


