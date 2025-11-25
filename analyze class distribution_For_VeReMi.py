# The libraries we need

import pandas as pd
import numpy as np

import os
import matplotlib.pyplot as plt

 

#Target_file_loc = r"E:\VeReMi Dataset\VeReMi_Extension Dataset for Misbehaviors in VANETs\mixalldata_clean.csv"

#Target_file_loc =  r"E:\VeReMi Dataset\VeReMi and BSMList\Main_data_shuffled.csv"

#Target_file_loc = r"E:\VeReMi Dataset\VeReMi and BSMList\Main_data_shuffled_CIC_Features_to_binary.csv" 

#Target_file_loc = r"E:\VeReMi Dataset\Some VeReMi files\traceJSON-9-7-A0-1-0.csv"

#Target_file_loc = r"E:\VeReMi Dataset\Some VeReMi files\traceJSON-9-7-A0-1-0_deepseek.csv"

Target_file_loc = r"E:\VeReMi Dataset\VeReMi from Zenodo\balanced_veremi_dataset.csv"

 
#Target_file_loc = r"E:\VeReMi Dataset\VeReMi and BSMList\Main_data_shuffled_CIC_Features_to_binary_under.csv"

## Reading a CSV file with low_memory set to False
Data_target_df = pd.read_csv(Target_file_loc, low_memory=False)


Data_target_df.info()

Data_target_df.head()
Data_target_df.dtypes

print(Data_target_df)

print(" >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>. .. .. .. .")

#get file size  and # Conversion to kilobytes, megabytes
file_size_bytes = os.path.getsize(Target_file_loc)
file_size_kb = file_size_bytes / 1024
file_size_mb = file_size_kb / 1024


print("File Size is :", file_size_mb, "MB")


#The problem is with csv files in CICDDos 2019, there are leading whistespaces in column names, because of which the key error is coming.
Data_target_df.columns = Data_target_df.columns.str.strip()

print(Data_target_df.describe())

#print("analyze class distribution ", Data_target_df.groupby("attacktype").size())    ##   VeReMi and BSMList .. .  From Zenodo 6 Mar 2024 .. . 
#print("analyze class distribution ", Data_target_df.groupby("hazardAttack").size())    ##   VeReMiAP

print("analyze class distribution ", Data_target_df.groupby("AttackerType").size())    ##   VeReMi  .. .  From Zenodo 21 Feb 2025 .. .
# Zenodo 21 Feb 2025 at link:  https://zenodo.org/records/14903687


print(" **************************************")

# Get all column names
column_names = Data_target_df.columns

# Print the column names
print(column_names)

#  Check existing index labels
print(Data_target_df.index)

#y = Data_target_df['attacktype']  
#y = Data_target_df['hazardAttack']  
y = Data_target_df['AttackerType']  


print('\n \n AttackerType_info= \n' , y.info())
print('\n \n AttackerType_head= \n' , y.head())






