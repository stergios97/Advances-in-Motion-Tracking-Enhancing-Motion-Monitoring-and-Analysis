##############################################################
#                                                            #
#    Code in this file is based on code aquired from:        #
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 7                                               #
#                                                            #
##############################################################

import numpy as np
import pandas as pd
from pathlib import Path
import time
start = time.time()

from Chapter7.PrepareDatasetForLearning import PrepareDatasetForLearning
from Chapter7.LearningAlgorithms import ClassificationAlgorithms
from Chapter7.Evaluation import ClassificationEvaluation
from util import util
from util.VisualizeDataset import VisualizeDataset

import pandas as pd
import scipy.linalg
import copy
import random
import numpy as np
from scipy import linalg
import inspyred
from Chapter8.dynsys.Model import Model
from Chapter8.dynsys.Evaluator import Evaluator
from pybrain.structure import RecurrentNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, FullConnection
from pybrain.datasets import SequentialDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.supervised.trainers import RPropMinusTrainer
from pybrain.tools.validation import testOnSequenceData
from pybrain.tools.shortcuts import buildNetwork
from Chapter7.Evaluation import ClassificationEvaluation
from Chapter7.Evaluation import RegressionEvaluation
import sys
import matplotlib.pyplot as plot
import pyflux as pf
from statsmodels.tsa.arima_model import ARIMA

# Read the result from the previous chapter, and make sure the index is of the type datetime.
DATASET_PATH = Path('./intermediate_datafiles/125')
N_FORWARD_SELECTION = 50

dataset = pd.read_csv(DATASET_PATH / 'dataset_result_fe.csv', index_col=0)
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

learner = ClassificationAlgorithms()
eval = ClassificationEvaluation()

possible_feature_sets = [basic_features, basic_time, basic_freq, basic_time_freq, selected_features]
feature_names = ['initial set', 'Initial + time domain', 'Initial + frequency domain', 'Initial + time + frequency', 'Selected features']
N_KCV_REPEATS = 5


print('Preprocessing done. Took', time.time()-start, 'seconds.')

scores_over_all_algs = []

for i in range(0, len(possible_feature_sets)):
    selected_train_X = train_X[possible_feature_sets[i]]
    selected_test_X = test_X[possible_feature_sets[i]]

    # First we run our non deterministic classifiers a number of times to average their score.

    performance_tr_nn = 0
    performance_tr_rf = 0
    performance_tr_svm = 0
    performance_te_nn = 0
    performance_te_rf = 0
    performance_te_svm = 0

    for repeat in range(0, N_KCV_REPEATS):
        print("Training NeuralNetwork run {} / {} ... ".format(repeat, N_KCV_REPEATS, feature_names[i]))
        class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.feedforward_neural_network(
            selected_train_X, train_y, selected_test_X, gridsearch=True
        )
        print("Training RandomForest run {} / {} ... ".format(repeat, N_KCV_REPEATS, feature_names[i]))
        performance_tr_nn += eval.accuracy(train_y, class_train_y)
        performance_te_nn += eval.accuracy(test_y, class_test_y)
        
        class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.random_forest(
            selected_train_X, train_y, selected_test_X, gridsearch=True
        )
        
        performance_tr_rf += eval.accuracy(train_y, class_train_y)
        performance_te_rf += eval.accuracy(test_y, class_test_y)

        print("Training SVM run {} / {}, featureset: {}... ".format(repeat, N_KCV_REPEATS, feature_names[i]))
      
        class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.support_vector_machine_with_kernel(
            selected_train_X, train_y, selected_test_X, gridsearch=True
        )
        performance_tr_svm += eval.accuracy(train_y, class_train_y)
        performance_te_svm += eval.accuracy(test_y, class_test_y)

    
    overall_performance_tr_nn = performance_tr_nn/N_KCV_REPEATS
    overall_performance_te_nn = performance_te_nn/N_KCV_REPEATS
    overall_performance_tr_rf = performance_tr_rf/N_KCV_REPEATS
    overall_performance_te_rf = performance_te_rf/N_KCV_REPEATS
    overall_performance_tr_svm = performance_tr_svm/N_KCV_REPEATS
    overall_performance_te_svm = performance_te_svm/N_KCV_REPEATS

# Run the deterministic classifiers:
    print("Determenistic Classifiers:")

    print("Training Nearest Neighbor run 1 / 1, featureset {}:".format(feature_names[i]))
    class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.k_nearest_neighbor(
        selected_train_X, train_y, selected_test_X, gridsearch=True
    )
    performance_tr_knn = eval.accuracy(train_y, class_train_y)
    performance_te_knn = eval.accuracy(test_y, class_test_y)
    print("Training Descision Tree run 1 / 1  featureset {}:".format(feature_names[i]))
    class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.decision_tree(
        selected_train_X, train_y, selected_test_X, gridsearch=True
    )
    
    performance_tr_dt = eval.accuracy(train_y, class_train_y)
    performance_te_dt = eval.accuracy(test_y, class_test_y)
    print("Training Naive Bayes run 1/1 featureset {}:".format(feature_names[i]))
    class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.naive_bayes(
        selected_train_X, train_y, selected_test_X
    )
   
    performance_tr_nb = eval.accuracy(train_y, class_train_y)
    performance_te_nb = eval.accuracy(test_y, class_test_y)

    scores_with_sd = util.print_table_row_performances(feature_names[i], len(selected_train_X.index), len(selected_test_X.index), [
                                                                                                (overall_performance_tr_nn, overall_performance_te_nn),
                                                                                                (overall_performance_tr_rf, overall_performance_te_rf),
                                                                                                (overall_performance_tr_svm, overall_performance_te_svm),
                                                                                                (performance_tr_knn, performance_te_knn),
                                                                                                (performance_tr_dt, performance_te_dt),
                                                                                                (performance_tr_nb, performance_te_nb)])
    scores_over_all_algs.append(scores_with_sd)

scores_over_all_algs_np = np.array(scores_over_all_algs)
np.save('scores_over_all_algs.npy', scores_over_all_algs_np)
DataViz.plot_performances_classification(['NN', 'RF','SVM', 'KNN', 'DT', 'NB'], feature_names, scores_over_all_algs)

class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.random_forest(
    train_X[selected_features], train_y, test_X[selected_features],
    gridsearch=True, print_model_details=True)

test_cm = eval.confusion_matrix(test_y, class_test_y, class_train_prob_y.columns)

DataViz.plot_confusion_matrix(test_cm, class_train_prob_y.columns, normalize=False)