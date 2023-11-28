##############################################################
#                                                            #
#    Code in this file is based on code aquired from:        #
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 4                                               #
#                                                            #
##############################################################

import sys
import copy
import pandas as pd
import time
from pathlib import Path
import argparse

from util.VisualizeDataset import VisualizeDataset
from Chapter4.TemporalAbstraction import NumericalAbstraction
from Chapter4.TemporalAbstraction import CategoricalAbstraction
from Chapter4.FrequencyAbstraction import FourierTransformation
from Chapter4.TextAbstraction import TextAbstraction

# Read the result from the previous chapter, and make sure the index is of the type datetime.
DATA_PATH = Path('./intermediate_datafiles/125/')
DATASET_FNAME = 'dataset_result_lowpass.csv'

def main():
    start_time = time.time()
    try:
        dataset = pd.read_csv(DATA_PATH / DATASET_FNAME, index_col=0)
        dataset.index = pd.to_datetime(dataset.index)
    except IOError as e:
        print('File not found, try to run previous crowdsignals scripts first!')
        raise e

    # Let us create our visualization class again.
    DataViz = VisualizeDataset(__file__)

    # Compute the number of milliseconds covered by an instance based on the first two rows
    milliseconds_per_instance = (dataset.index[1] - dataset.index[0]).microseconds/1000

    NumAbs = NumericalAbstraction()
    FreqAbs = FourierTransformation()
    
    periodic_measurements = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 
                             'linear_x', 'linear_y', 'linear_z', 'mag_x', 'mag_y', 'mag_z']


    if FLAGS.mode == 'aggregation':
        window_sizes = [int(float(3*1000)/milliseconds_per_instance), int(float(15*1000)/milliseconds_per_instance), int(float(2*60*1000)/milliseconds_per_instance)]   
        for col in periodic_measurements:
            for ws in window_sizes:
                dataset = NumAbs.abstract_numerical(dataset, [col], ws, 'mean')
                dataset = NumAbs.abstract_numerical(dataset, [col], ws, 'std')

            DataViz.plot_dataset(dataset, [col, col+'_temp_mean'], ['exact', 'like'], ['line', 'line'])
        
    if FLAGS.mode == 'aggregation_frequency':
        
        fs = float(1000)/milliseconds_per_instance
        ws = int(float(15*1000)/milliseconds_per_instance)
        
        features = ['mean', 'median', 'std', 'min', 'max']
        for feature in features: 
            print('Creating aggregated feature', feature)
            dataset = NumAbs.abstract_numerical(dataset, periodic_measurements, ws, feature)
        
        print('Creating fequency features')
        dataset = FreqAbs.abstract_frequency(copy.deepcopy(dataset), periodic_measurements, int(float(2.5*1000)/milliseconds_per_instance), fs)
        DataViz.plot_dataset(dataset, ['acc_x_max_freq', 'acc_x_freq_weighted', 'acc_x_pse', 'activity'], ['like', 'like', 'like', 'like'], ['line', 'line', 'line','points'])

        # The percentage of overlap we allow
        window_overlap = 0.7
        skip_points = int((1-window_overlap) * ws)
        dataset = dataset.iloc[::skip_points,:]
        print('Size of final dataset is', dataset.shape)
        print("--- %s seconds ---" % (time.time() - start_time))
        dataset.to_csv(DATA_PATH / 'dataset_result_fe.csv')
    
if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='aggregation_frequency',
                        help= "Select what mode to run", choices=['aggregation_frequency', 'aggregation']) 

    FLAGS, unparsed = parser.parse_known_args()
    
    main()