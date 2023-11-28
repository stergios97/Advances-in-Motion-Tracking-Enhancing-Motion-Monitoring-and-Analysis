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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from util.VisualizeDataset import VisualizeDataset
from util import util
from pathlib import Path
# Read the dataset
DATASET_PATH = Path('./intermediate_datafiles/')
df = pd.read_csv(DATASET_PATH / 'preprocessed_dataset_125.csv', index_col=0)
df.index = pd.to_datetime(df.index)

# Plot the data
DataViz = VisualizeDataset(__file__)

print(df)

# Get an overview of the dataset
print(df.info())

# Number of rows & columns
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print(df.describe())

# Calculate the proportion of missing values for each attribute
missing_proportions = df.iloc[:, :-6].isnull().mean()  # Exclude the last 6 columns(labels)
print(missing_proportions)


### VISUALIZATION ###

# A bar plot to visualize missing proportions
plt.figure(figsize=(10, 6))  # size of the plot
missing_proportions.plot(kind='bar')  # plotting the proportion of missing values as bars

# Customization of the plot
plt.title('Missing Proportions', fontsize=14)
plt.xlabel('Attributes', fontsize=14)
plt.ylabel('Proportion', fontsize=14)
plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability

plt.tight_layout()

plt.show()

# Combine 'acc_x', 'acc_y', and 'acc_z' into a single attribute
df['combined_acc'] = df[['acc_x', 'acc_y', 'acc_z']].mean(axis=1)

# Set the colors for each accelerometer attribute
colors = ['orange', 'purple', 'blue']

# Create a figure and axes
fig, ax = plt.subplots()

# Plot the combined accelerometer data with specified colors
for i, col in enumerate(['acc_x', 'acc_y', 'acc_z']):
    ax.plot(df[col], color=colors[i], label=col)

# Set the title and labels for the graph
ax.set_title('Combined Accelerometer Data', fontsize=14)
ax.set_xlabel('Time', fontsize=14)
ax.set_ylabel('Acceleration', fontsize=14)
ax.legend()
plt.show()

# Plot all data
DataViz.plot_dataset(df, ['acc_', 'gyr_', 'linear_', 'mag_', 'prox_', 'loc_', 'activity'],
                                ['like', 'like', 'like', 'like', 'like', 'like', 'like'],
                                ['line', 'line', 'line', 'line', 'line', 'line', 'points'])


# Distribution analysis
sensor_columns = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z',
                  'linear_x', 'linear_y', 'linear_z', 'mag_x', 'mag_y', 'mag_z']

fig, ax = plt.subplots(figsize=(10, 6))

# Box-plots for each sensor column
boxplot_data = []
for column in sensor_columns:
    boxplot_data.append(df[column])

ax.boxplot(boxplot_data, labels=sensor_columns)

# ax.set_title('Box Plot of Sensor Readings', fontsize=14)
ax.set_xlabel('Sensor', fontsize=12)
ax.set_ylabel('Value', fontsize=12)
plt.xticks(rotation=45)

plt.show()


fig, ax = plt.subplots(figsize=(10, 6))

violinplot_data = [df[column] for column in sensor_columns]

# Violin plots for each sensor column
ax.violinplot(violinplot_data)

ax.set_xticks(range(1, len(sensor_columns) + 1))
ax.set_xticklabels(sensor_columns, rotation=45)
ax.set_title('Violin Plot of Sensor Readings', fontsize=14)
ax.set_xlabel('Sensor', fontsize=12)
ax.set_ylabel('Value', fontsize=12)


plt.show()