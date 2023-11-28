import numpy as np
import pandas as pd
from pathlib import Path
import copy
import optuna
from Chapter7.PrepareDatasetForLearning import PrepareDatasetForLearning
from sklearn.preprocessing import LabelEncoder
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.utils import to_categorical 

# Read the result from the previous chapter, and make sure the index is of the type datetime.
DATASET_PATH = Path('./intermediate_datafiles/125')

dataset = pd.read_csv(DATASET_PATH / 'dataset_result_lowpass.csv', index_col=0)
dataset.index = pd.to_datetime(dataset.index)


activityColumns = dataset.filter(regex='^activity').columns.tolist()
label_encoder = LabelEncoder()
dataset['activity'] = label_encoder.fit_transform(dataset[activityColumns].idxmax(axis=1))
dataset.drop(activityColumns, axis=1, inplace=True)
dataset = dataset.dropna()

prepare = PrepareDatasetForLearning()

train_X, test_X, train_y, test_y = prepare.split_single_dataset_classification(dataset, ['activity'], 
                                                                               'exact', 0.8, filter=False, temporal=True)

train_y = train_y.values
test_y = test_y.values

# Initial features acquired from just the raw dataset
basic_features = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z',
       'linear_x', 'linear_y', 'linear_z', 'loc_Latitude', 'loc_Longitude',
       'loc_Height', 'loc_Velocity', 'loc_Direction', 'loc_Horizontal',
       'loc_Vertical', 'mag_x', 'mag_y', 'mag_z', 'prox_Distance']

time_features = [name for name in dataset.columns if '_temp_' in name]
freq_features = [name for name in dataset.columns if (('_freq' in name) or ('_pse' in name))]
print('#initial features: ', len(basic_features))

print('#time features: ', len(time_features))
print('#frequency features: ', len(freq_features))

basic_time = list(set().union(basic_features, time_features))
basic_freq = list(set().union(basic_features, freq_features))
basic_time_freq = list(set().union(basic_features, freq_features, time_features))

# Features selected by the feature selection 
selected_features = [
 'acc_z_temp_min_ws_120',
 'acc_y_temp_min_ws_120',
 'linear_z_temp_median_ws_120',
 'mag_y_temp_max_ws_120',
 'loc_Longitude',
 'mag_y',
 'gyr_z_freq_4.0_Hz_ws_20',
 'acc_y_freq_1.2_Hz_ws_20',
 'gyr_z_freq_1.2_Hz_ws_20',
 'gyr_y_pse',
 'linear_z_freq_1.2_Hz_ws_20',
 'linear_y_freq_2.0_Hz_ws_20',
 'linear_x_freq_0.0_Hz_ws_20'
]

possible_feature_sets = [basic_features, basic_time, basic_freq, basic_time_freq, selected_features]
feature_names = ['initial set', 'Initial + time domain', 'Initial + frequency domain', 'Initial + time + frequency', 'Selected features']

features = basic_features

train_x = train_X[features]
test_x = test_X[features]

def create_numerical_multiple_dataset(train, test):

    # Combine the two datasets as we want to include all possible values
    # for the categorical attribute.
    total_dataset = pd.concat([train, test])

    # Convert and split up again.
    total_dataset = pd.get_dummies(pd.DataFrame(total_dataset), prefix='', prefix_sep='')
    new_train = copy.deepcopy(total_dataset.iloc[0:len(train.index),:])
    new_test = copy.deepcopy(total_dataset.iloc[len(train.index):len(train.index)+len(test.index),:])
    return new_train, new_test

def normalize(train, test, range_min, range_max):

    total = pd.concat([copy.deepcopy(train), test], ignore_index=True)

    max = total.max()
    min = total.min()
    difference = max - min
    difference = difference.replace(0, 1)

    new_train = (((train - min)/difference) * (range_max - range_min)) + range_min
    new_test = (((test - min)/difference) * (range_max - range_min)) + range_min
    return new_train, new_test, min, max


def denormalize(y, min, max, range_min, range_max):
    difference = max - min
    difference = difference.replace(0, 1)

    y = (y - range_min)/(range_max - range_min)

    return (y * difference) + min


def model_lstm(train_x, train_y, test_x, test_y, hyperparameters):
    # Define the LSTM model
    print('Creating model')
    inputshape = (train_x.shape[1:])
    model = Sequential()
    model.add(LSTM(units=hyperparameters['lstm_units'], input_shape=inputshape, kernel_regularizer=l2(hyperparameters['l2'])))
    model.add(Dropout(hyperparameters['dropout_rate']))
    model.add(Dense(units=train_y.shape[1], activation='softmax', kernel_regularizer=l2(hyperparameters['l2'])))

    print('Compiling model')
    # Compile the model
    optimizer = Adam(learning_rate=hyperparameters['learning_rate'])
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Define early stopping
    early_stopping = EarlyStopping(patience=hyperparameters['patience'], restore_best_weights=True)

    print('Fitting model')
    model.fit(train_x, train_y, epochs=hyperparameters['epochs'],
              batch_size=hyperparameters['batch_size'], callbacks=[early_stopping], 
              verbose=0, validation_data=(test_x, test_y))

    print('Evaluating model')
    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(test_x, test_y)

    return accuracy


time_step_trials = [4, 6, 8, 10]
results_time_steps = []
for TIME_STEPS in time_step_trials:
    print('Starting trials for time steps =', TIME_STEPS)
    print('Building training and test data')
    new_train_X, new_test_X = create_numerical_multiple_dataset(train_x, test_x)
    new_train_X, new_test_X, min_X, max_X = normalize(new_train_X, new_test_X, 0, 1)

    # Convert the target variable to one-hot encoded vectors
    new_train_y = to_categorical(train_y)
    new_test_y = to_categorical(test_y)

    print('Making training data sequential')
    # Split the data into input sequences
    train_x_sequences = []
    train_y_sequences = []
    for i in range(len(new_train_X) - TIME_STEPS + 1):
        train_x_sequences.append(new_train_X[i:i+TIME_STEPS])
        train_y_sequences.append(new_train_y[i+TIME_STEPS-1])
    train_x_sequences = np.array(train_x_sequences)
    train_y_sequences = np.array(train_y_sequences)

    print('Making test data sequential')
    test_x_sequences = []
    test_y_sequences = []
    for i in range(len(new_test_X) - TIME_STEPS + 1):
        test_x_sequences.append(new_test_X[i:i+TIME_STEPS])
        test_y_sequences.append(new_test_y[i+TIME_STEPS-1])
    test_x_sequences = np.array(test_x_sequences)
    test_y_sequences = np.array(test_y_sequences)

    def objective(trial):
        # Define hyperparameters to search
        hyperparameters = {
            'lstm_units': trial.suggest_int('lstm_units', 8, 64),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'epochs': 20,
            'batch_size': 64,
            'patience': 5,
            'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.2),
            'l2':  trial.suggest_float('l2', 0.001, 0.15),
        }

        # Build the LSTM model
        accuracy = model_lstm(train_x_sequences, train_y_sequences, test_x_sequences, test_y_sequences, hyperparameters)

        return accuracy

    print('Starting optimization')
    # Create Optuna study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    # Get the best hyperparameters
    best_hyperparameters = study.best_params
    best_accuracy = study.best_value

    print("Best Hyperparameters:", best_hyperparameters)
    print("Best Accuracy:", best_accuracy)
    res = {'hyperparameters': best_hyperparameters,
           'accuracy': best_accuracy}
    results_time_steps.append(res)

import pickle

file_path = 'tuning_results.pkl'

# Save the dictionary to a pickle file
with open(file_path, 'wb') as f:
    pickle.dump(results_time_steps, f)

