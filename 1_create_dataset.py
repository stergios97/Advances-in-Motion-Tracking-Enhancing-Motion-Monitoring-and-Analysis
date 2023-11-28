##############################################################
#                                                            #
#    Code in this file is based on code aquired from:        #
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 2                                               #
#                                                            #
##############################################################

from Chapter2.CreateDataset import CreateDataset
from pathlib import Path

DATASET_PATH = Path('./datasets/collected/')
RESULT_PATH = Path('./intermediate_datafiles/')

# Set a granularity (the discrete step size of our time series data). We'll use a course-grained granularity of one
# instance per minute, and a fine-grained one with four instances per second.
GRANULARITIES = [125]

# We can call Path.mkdir(exist_ok=True) to make any required directories if they don't already exist.
[path.mkdir(exist_ok=True, parents=True) for path in [DATASET_PATH, RESULT_PATH]]

datasets = []

for granularity in GRANULARITIES:
    print(f'Creating numerical datasets from files in {DATASET_PATH} using granularity {granularity}.')

    # Create an initial dataset object with the base directory for our data and a granularity
    dataset = CreateDataset(DATASET_PATH, granularity)

    # Add the selected measurements to it.
    dataset.add_numerical_dataset('Accelerometer.csv', 'Time', ['x','y','z'], 'avg', 'acc_')
    dataset.add_numerical_dataset('Gyroscope.csv', 'Time', ['x','y','z'], 'avg', 'gyr_')
    dataset.add_numerical_dataset('Linear Acceleration.csv', 'Time', ['x','y','z'], 'avg', 'linear_')
    dataset.add_numerical_dataset('Location.csv', 'Time', ['Latitude', 'Longitude', 'Height', 'Velocity', 'Direction', 'Horizontal', 'Vertical'], 'avg', 'loc_')
    dataset.add_numerical_dataset('Magnetometer.csv', 'Time', ['x','y','z'], 'avg', 'mag_')
    dataset.add_numerical_dataset('Proximity.csv', 'Time', ['Distance'], 'avg', 'prox_')

    # We add the labels provided by the users. These are categorical events that might overlap. We add them
    dataset.add_event_dataset('Labels.csv', 'start_time', 'end_time', 'activity', 'binary')

    dataset = dataset.data_table
    dataset.to_csv(RESULT_PATH / ('preprocessed_dataset' + '_' + str(granularity) + '.csv'))
print('The code has run through successfully!')