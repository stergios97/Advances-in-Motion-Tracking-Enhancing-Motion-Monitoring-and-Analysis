import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from Chapter7.PrepareDatasetForLearning import PrepareDatasetForLearning
from util.VisualizeDataset import VisualizeDataset
from util import util
from pathlib import Path


# Load the dataset
DATASET_PATH = Path('./intermediate_datafiles/')
DATASET_NAME = 'dataset_result_fe.csv'

#Read the dataset
dataset = pd.read_csv(DATASET_PATH / DATASET_NAME)
dataset.index = pd.to_datetime(dataset.index)

# Create one target column
activityColumns = dataset.filter(regex='^activity').columns.tolist()
dataset['activity'] = dataset[activityColumns].idxmax(axis=1)
dataset.drop(activityColumns, axis=1, inplace=True)
dataset = dataset.dropna()

DataViz = VisualizeDataset(__file__)

prepare = PrepareDatasetForLearning()

train_X, test_X, train_y, test_y = prepare.split_single_dataset_classification(dataset, ['activity'], 
                                                                               'exact', 0.7, filter=False, temporal=False)

print('Training set length is: ', len(train_X.index))
print('Test set length is: ', len(test_X.index))


# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(train_X, train_y, epochs=100, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(train_X)
testPredict = model.predict(test_X)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
train_y = scaler.inverse_transform([train_y])
testPredict = scaler.inverse_transform(testPredict)
test_y = scaler.inverse_transform([test_y])

# calculate root mean squared error
trainScore = np.sqrt(mean_squared_error(train_y[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = np.sqrt(mean_squared_error(test_y[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# # shift train predictions for plotting
# trainPredictPlot = np.empty_like(dataset)
# trainPredictPlot[:, :] = np.nan
# trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# # shift test predictions for plotting
# testPredictPlot = np.empty_like(dataset)
# testPredictPlot[:, :] = np.nan
# testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# # plot baseline and predictions
# plt.plot(scaler.inverse_transform(dataset))
# plt.plot(trainPredictPlot)
# plt.plot(testPredictPlot)
# plt.show()
