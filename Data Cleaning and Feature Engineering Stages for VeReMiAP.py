#This code excute Data Cleaning and Feature Engineering Stages .. . 

# The libraries we need .. . 


import os

import pandas as pd
import numpy as np

# The libraries we need .. . for Handling Missing Data with Mixed Feature Types By KNN .. .

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
### 

from sklearn.ensemble import IsolationForest


def is_float(value):
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False

    
#Target_file_loc  = r"E:\VeReMi Dataset\VeReMi and BSMList\Main_data_shuffled_CIC_Features.csv"
Target_file_loc = r"E:\VeReMi Dataset\Some VeReMi files\traceJSON-9-7-A0-1-0.csv" 
 

# **************************************  

## Reading a CSV file with low_memory set to False
#Data_target_df = pd.read_csv(Target_file_loc)
#Data_target_df = pd.read_csv(Target_file_loc, error_bad_lines=False)   # Source - https://stackoverflow.com/a

#Data_target_df = pd.read_csv(Target_file_loc , on_bad_lines='skip')


Data_target_df = pd.read_csv(Target_file_loc, low_memory=False)
 

Data_target_df.info()  



print(" >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>. .. .. .. .")

#get file size  # Conversion to kilobytes, megabytes  .. .

file_size_bytes = os.path.getsize(Target_file_loc)
file_size_kb = file_size_bytes / 1024
file_size_mb = file_size_kb / 1024


print("Sample Size is :", file_size_mb, "MB")

#The problem is with csv files in CICDDos 2019, there are leading whistespaces in column names, because of which the key error is coming .. .

Data_target_df.columns = Data_target_df.columns.str.strip()




#Data Cleaning and Feature Engineering

#There are some columns that are not really useful and hence we will proceed to drop them.
#Also, there are some missing values so let’s drop all those rows with empty values:

print(Data_target_df.info())


print(" **************************************")


print("DataFrame  after modified  >>> ")

Data_target_df.head()

numerical_features = ['EventID', 'exchangedMessageType', 'hazardAttack', 'hazardOccurrence', 'hazardOccurrencePercentage', 'laneIndex', 'lanePosition', 'maxDeceleration', 'maxSpeed' , 'messageID' , 'rcvTime', 'sendTime' , 'sender' , 'senderPseudo' , 'type', 'vehicleId' ]
categorical_features = ['EventType', 'RoadID', 'acl', 'acl_noise', 'currentDirection', 'hed', 'hed_noise', 'pos' , 'pos_noise' , 'sender_GPS' , 'spd' , 'spd_noise']


# Handling "EventType" column if "Accident": 
# Convert values ​​'Accident' to 1 and empty values ​​to 0:
Data_target_df['EventType'] = Data_target_df['EventType'].apply(lambda x: 1 if x == 'Accident' else (0 if pd.isna(x) or x == '' or x is None else x))
# If you want to convert the entire column to integers (optional):
#Data_target_df['EventType'] = pd.to_numeric(Data_target_df['EventType'], errors='coerce').fillna(0).astype(int)


# The noisy data rectifying step:
#Removing duplicate records can help reduce noise and redundancy in our dataset.

# Remove duplicate rows:  
Data_target_df = Data_target_df.drop_duplicates()


# Remove anomaly rows:  
# Delete the rows containing float values ​​in the 'vehicleId' column .. .
mask = Data_target_df['vehicleId'].apply(lambda x: not is_float(x) or float(x).is_integer())
Data_target_df = Data_target_df[mask]


# Handling missing values ​​using KNN:
# Pipeline for numerical features
numerical_pipeline = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=2)),  # Apply KNN Imputer first
    ('scaler', StandardScaler())
])

# Pipeline for categorical features
categorical_pipeline = Pipeline(steps=[
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),  # Encode categories first
    ('imputer', KNNImputer(n_neighbors=2))  # Apply KNN Imputer after encoding
])

# Combine pipelines into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])


# Applying the pipeline to the data
Data_target_df = preprocessor.fit_transform(Data_target_df)

'''
# Extracting the column names
num_cols = ['Numerical_1', 'Numerical_2']
cat_cols = ['Categorical']

# Combining the column names
columns = num_cols + cat_cols

# Convert the imputed data back to a DataFrame
df_imputed = pd.DataFrame(df_imputed, columns=columns)

print("Data after KNN Imputation with Mixed Feature Types:\n", df_imputed) '''
############ ###


'''
# Addressing noise using Isolation Forest
iso_forest = IsolationForest(contamination=0.05)
outliers = iso_forest.fit_predict(Data_target_df)


# Removing outliers:
Data_target_df = Data_target_df[outliers == 1]  

# The infinity data values rectifying step:
Data_target_df.replace([np.inf, -np.inf], np.nan)
Data_target_df.dropna(inplace=True)


# label encoding
from sklearn.preprocessing import LabelEncoder
for col in Data_target_df.columns:
    le = LabelEncoder()
    Data_target_df[col] = le.fit_transform(Data_target_df[col])
Data_target_df.info()

 '''

# save target dataframe to new location
'''Data_target_df.to_csv(
         r"E:\VeReMi Dataset\Some VeReMi files\jsonoutput-9-7-A0-1-0_cleaned.csv",
        index=False)
        '''

# Source - https://stackoverflow.com/a
# Posted by Elvin Aghammadzada
# Retrieved 2025-11-20, License - CC BY-SA 4.0

pd.DataFrame(Data_target_df).to_csv(r"E:\VeReMi Dataset\Some VeReMi files\jsonoutput-9-7-A0-1-0_cleaned.csv")

 



