# RNN-for-regression
Stock price prediction for 5 years

## Analysis
### Create the dataset
* For example, the target is the tenth day’s open price, so the features are the ninth day, the eighth day and the seventh day’s open, volume, high, low. So there are 12 features in one sample, and one target for each sample as well.  
* I use a for loop to create the data, to get the open, volume, high and low features of three days and get target open price for the fourth day.
  features = feature_Volume + feature_Open + feature_High + feature_Low. 
* For training dataset: 70% data, saved as train_data_RNN.csv. For testing dataset: 30% data, saved as test_data_RNN.csv. 

### Preprocessing steps
* Use the MinMaxScaler to reshape 12 features(input) and 1 target(output).  
* Reshape the input data into 3D: [the number of samples, timesteps, the number of features]. Because I use LSTM, and LSTM’s input is 3D.  

