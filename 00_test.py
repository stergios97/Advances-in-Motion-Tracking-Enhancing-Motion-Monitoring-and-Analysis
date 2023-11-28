from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
from util.VisualizeDataset import VisualizeDataset
from Chapter3.DataTransformation import LowPassFilter
from Chapter3.DataTransformation import PrincipalComponentAnalysis
from Chapter3.ImputationMissingValues import ImputationMissingValues
from Chapter3.KalmanFilters import KalmanFilters
import numpy as np

data = np.random.random((1810, 20))  # Replace with your actual array

# Reshape the array
reshaped_data = data.reshape((1810, 4, 20))