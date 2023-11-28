##############################################################
#                                                            #
#    Code in this file is based on code aquired from:        #
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 3                                               #
#                                                            #
##############################################################

import copy
from pathlib import Path
import pandas as pd
import argparse

from util.VisualizeDataset import VisualizeDataset
from Chapter3.DataTransformation import LowPassFilter
from Chapter3.DataTransformation import PrincipalComponentAnalysis
from Chapter3.ImputationMissingValues import ImputationMissingValues
from Chapter3.OutlierDetection import DistributionBasedOutlierDetection
from Chapter3.OutlierDetection import DistanceBasedOutlierDetection
from Chapter3.KalmanFilters import KalmanFilters

# Set up the file names and locations.
DATA_PATH = Path('./intermediate_datafiles/')
DATA_PATH_RESULT = DATA_PATH.joinpath('125/') 
DATASET_FNAME = 'preprocessed_dataset_125.csv'

def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():

    print_flags()

    # Next, import the data from the specified location and parse the date index.
    try:
        dataset = pd.read_csv(Path(DATA_PATH / DATASET_FNAME), index_col=0)
        dataset.index = pd.to_datetime(dataset.index)
    except IOError as e:
        print('File not found, try to run previous crowdsignals scripts first!')
        raise e

    # We'll create an instance of our visualization class to plot the results.
    DataViz = VisualizeDataset(__file__)

    # Compute the number of milliseconds covered by an instance based on the first two rows
    milliseconds_per_instance = (
        dataset.index[1] - dataset.index[0]).microseconds/1000

    MisVal = ImputationMissingValues()
    LowPass = LowPassFilter()
    PCA = PrincipalComponentAnalysis()
    
    # Check all columns for outliers
    outlier_columns = [col for col in dataset.columns if ('activity' not in col)]
    
    # Create the outlier classes.
    OutlierDistr = DistributionBasedOutlierDetection()
    OutlierDist = DistanceBasedOutlierDetection()

    if FLAGS.mode == 'mixture_imputation_lowpass':
        # Define the threshold for the mixture models
        THRESHOLD = 0.00001
        
        # Keep track of how many outliers we identified
        total_outlier = 0
        total_values = 0
        for col in outlier_columns:
            # Applying the mixture model
            print(f"Applying mixture model for column {col}")
            dataset = OutlierDistr.mixture_model(dataset, col, 6)
            # DataViz.plot_dataset(dataset, [
            #                      col, col + '_mixture_probability'], ['exact', 'exact'], ['line', 'points'], y_axis_titles=['m','probability'])
            
            # Compute the number of values that were flagged as outliers and update counts. 
            num_set_to_na = dataset.loc[dataset[col + '_mixture_probability'] < THRESHOLD, col].count()
            total_outlier += num_set_to_na
            total_values += len(dataset)
            print('{}/{} values detected as outlier for col {}'.format(num_set_to_na, len(dataset), col))
            
            # Set all outliers as missing values.
            dataset.loc[dataset[col + '_mixture_probability'] < THRESHOLD, col] = pd.NA
            dataset = dataset.drop([col + '_mixture_probability'], axis=1)
            
        
        # print(total_outlier, total_values)
        
        # # Save result to csv
        dataset.to_csv(DATA_PATH_RESULT / 'dataset_result_mixture.csv')
    
        # Get all columns with missing values
        na_columns = dataset.columns[dataset.isna().any()].tolist()
        na_columns = [col for col in na_columns if 'activity' not in col]
        
        # Impute all missing values using linear interpolation
        for col in na_columns:
            print(f"Imputing missing values for column {col}\n",  dataset[col].isnull().sum(), 'missing values')
            dataset = MisVal.impute_interpolate(dataset, col)
            # imputed_dataset = MisVal.impute_interpolate(copy.deepcopy(dataset), col)
            # DataViz.plot_imputed_values(dataset, ['original ' + col, 'interpolation ' + col], col,
            #                         imputed_dataset[col], y_axis_title= 'm')
            
        # Save result to csv
        dataset.to_csv(DATA_PATH_RESULT / 'dataset_result_interpolation.csv')
        
        # Define all features with periodic behaviour 
        periodic_measurements = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'linear_x', 'linear_y', 'linear_z', 'mag_x', 'mag_y', 'mag_z']
        
        # Determine the sampling frequency and define the cutoff point.
        fs = float(1000)/milliseconds_per_instance
        cutoff = 3

        # Apply lowpass filter to the specified columns
        for col in periodic_measurements:
            print('Applying Lowpass filter to column: ', col)
            dataset = LowPass.low_pass_filter(
                dataset, col, fs, cutoff, order=10)
            dataset[col] = dataset[col + '_lowpass']
            # DataViz.plot_dataset(dataset.iloc[int(0.22*len(dataset.index)):int(0.25*len(dataset.index)), :],
                            #  [col, col+ '_lowpass'], ['exact', 'exact'], ['line', 'line'])
            del dataset[col + '_lowpass']
            
        # Save result to csv    
        dataset.to_csv(DATA_PATH_RESULT / 'dataset_result_lowpass.csv')
        
    elif FLAGS.mode == 'kalman':
        KalFilter = KalmanFilters()
        
        # applying the kalman filter to all feature columns
        for col in [c for c in dataset.columns if not 'activity' in c]:
            print('Applying Kalman filter for column: ', col)
            dataset = KalFilter.apply_kalman_filter(dataset, col)
            # DataViz.plot_imputed_values(dataset, [
                                        # 'original', 'kalman'], col, dataset[col + '_kalman'])
            DataViz.plot_dataset(dataset, [col], ['like'], ['line'], y_axis_title= 'm/s^2')
        
        # Store the dataset. 
        dataset.to_csv(DATA_PATH / 'dataset_result_kalman.csv')
       
        DataViz.plot_dataset(dataset, ['acc_', 'gyr_', 'linear_', 'mag_', 'loc_', 'prox_', 'activity'],
                             ['like', 'like', 'like', 'like', 'like', 'like', 'like'],
                             ['line', 'line', 'line', 'line', 'line', 'line', 'points'])

    elif FLAGS.mode == 'PCA':

        # First impute, as PCA can not deal with missing values       
        for col in [c for c in dataset.columns if not 'label' in c]:
            dataset = MisVal.impute_interpolate(dataset, col)

       
        selected_predictor_cols = [c for c in dataset.columns if (
            not ('label' in c)) and (not (c == 'hr_watch_rate'))]
        pc_values = PCA.determine_pc_explained_variance(
            dataset, selected_predictor_cols)

        # Plot the variance explained.
        DataViz.plot_xy(x=[range(1, len(selected_predictor_cols)+1)], y=[pc_values],
                        xlabel='principal component number', ylabel='explained variance',
                        ylim=[0, 1], line_styles=['b-'])

        # We select 7 as the best number of PC's as this explains most of the variance

        n_pcs = 7

        dataset = PCA.apply_pca(copy.deepcopy(
            dataset), selected_predictor_cols, n_pcs)

        # And we visualize the result of the PC's
        DataViz.plot_dataset(dataset, ['pca_', 'label'], [
                             'like', 'like'], ['line', 'points'])


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='mixture_imputation_lowpass',
                        help="Select what version to run: final, imputation, lowpass or PCA \
                        'lowpass' applies the lowpass-filter to a single variable \
                        'imputation' is used for the next chapter \
                        'PCA' is to study the effect of PCA and plot the results\
                        'final' is used for the next chapter", choices=['lowpass', 'kalman', 'mixture_imputation_lowpass', 'PCA', 'final'])

   
    FLAGS, unparsed = parser.parse_known_args()

    main()
