import cv2
import imutils
import numpy as np
import fnmatch
import os
import logging
import pandas as pd



dataset_path = '/Users/petesmac/Documents/Machine Learning/DATA/LEGO-brick-images/full_parts.csv'

df = pd.read_csv(dataset_path, encoding='utf-8', index_col='part_num')

dataset_files = [str(f)  for f in df.index]


print (len(dataset_files))
count=0
for partn in dataset_files:
    
    if partn.isnumeric() == True:
        count=count+1
print(count)
