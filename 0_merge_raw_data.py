from pathlib import Path
import pandas as pd
import os
from datetime import datetime, timedelta

USERS_PATH = Path('./datasets/collected/')
RESULT_PATH = Path('./intermediate_datafiles/')

START_TIME = datetime(2023, 6, 1, 12, 0)

COL_NAMES = {
    'Accelerometer.csv': ['Time', 'x', 'y', 'z'],
    'Gyroscope.csv': ['Time', 'x', 'y', 'z'],
    'Light.csv': ['Time', 'Illuminance'],
    'Linear Acceleration.csv': ['Time', 'x', 'y', 'z'],
    'Location.csv': ['Time', 'Latitude', 'Longitude', 'Height', 'Velocity', 'Direction', 'Horizontal', 'Vertical'],
    'Magnetometer.csv': ['Time', 'x', 'y', 'z'],
    'Proximity.csv': ['Time', 'Distance'],
    }
FILES = ['Accelerometer.csv', 'Gyroscope.csv', 'Linear Acceleration.csv', 'Location.csv', 'Magnetometer.csv', 'Proximity.csv']

# Find all user directories
users = [name for name in os.listdir(USERS_PATH) if os.path.isdir(os.path.join(USERS_PATH, name))]
print('Found users:', users, '\n')

datafiles = [[]] * (len(FILES) + 1)
time = 0

for user in users:
    print('Starting merge for user:', user)
    DATASET_PATH = USERS_PATH.joinpath(user)
    
    # Find all recorded activities (directories)
    print('Searching directories...')
    dirs = [name for name in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, name))]
    print('Found directories for the following activities: ', dirs)

    print('Merging files...')
    
    label_df = pd.DataFrame(columns=['start_time', 'end_time', 'activity'])
    date = START_TIME + timedelta(seconds=time)
    add_time = 0
    
    # Loop over all sensors to merge sensor data
    for idx, file in enumerate(FILES):
        result_df = []
        time_loc = time
        for dir in dirs:       
            temp_df = pd.read_csv(DATASET_PATH.joinpath(dir, file))
            
            # Check if the recorded activity is 300s or 600s. 
            activity_length = max(temp_df.iloc[:,0])
            activity_length = 300  if abs(activity_length-300) < abs(activity_length-600) else 600
            
            # Get rid of measurements that exceeded the max activity time
            temp_df = temp_df[temp_df.iloc[:, 0] <= activity_length]
            
            # Convert the time column to Unix timestamps
            temp_df.iloc[:,0] += time_loc
            temp_df.isetitem(0, ((START_TIME + pd.to_timedelta(temp_df.iloc[:,0], unit='s')).apply(datetime.timestamp) * 1000000000).astype('Int64'))
            
            # Check if there are multiple recordings for this activity. If so merge with existing dataframe
            if len(result_df) > 0:
                result_df = pd.concat([result_df, temp_df], ignore_index=True)
            else:
                result_df = temp_df
                
            # Adjust the time to create one long timeline    
            time_loc += activity_length
            
            # Create the dataframe that contains the timestamps for the activities (target values)
            if idx == 0:
                activity = dir.split('-')[0]
                new_row = [int(date.timestamp() * 1000000000), int((date + timedelta(seconds=activity_length)).timestamp() * 1000000000), activity]
                label_df.loc[len(label_df)] = new_row
                date += timedelta(seconds=activity_length)
                add_time += activity_length
            
        result_df.columns = COL_NAMES[file]
        
        # Merge the dataframe of this user with the others in the main dataframe
        if len(datafiles[idx]) > 0:
            datafiles[idx] = pd.concat([datafiles[idx], result_df], ignore_index=True)
        else:
            datafiles[idx] = result_df
        
    # Merge the files     
    if len(datafiles[-1]) > 0:
        datafiles[-1] = pd.concat([datafiles[-1], label_df], ignore_index=True)
    else:
        datafiles[-1] = label_df
    label_df.to_csv(DATASET_PATH.joinpath('Labels.csv'), index=False)
    time += add_time
    
# Convert all float64 columns to float16 to save memory and computation time in future algorithms.    
for idx, file in enumerate(FILES):
    float_cols = datafiles[idx].select_dtypes(include=['float64']).columns
    datafiles[idx][float_cols] = datafiles[idx][float_cols].astype('float16')
    # Save resulting dataframe to file
    datafiles[idx].to_csv(USERS_PATH.joinpath(file), index=False)
    
datafiles[-1].to_csv(USERS_PATH.joinpath('Labels.csv'), index=False)
print('Files merged!\n')