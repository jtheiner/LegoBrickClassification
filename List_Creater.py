import cv2
import imutils
import numpy as np
import fnmatch
import os
import logging
import pandas as pd
import re

dataset_path = './DATA/LEGO-brick-images/'
Ldraw_path = '../LegoBrickClassification/'
part_list_csv ='Parts_list.csv'


col_names =  ['Directory', 'Part_num']
parts_list  = pd.DataFrame(columns = col_names)
label_names = [str(filenames) for  filenames in fnmatch.filter(os.listdir(Ldraw_path+'ldraw/parts'), '*.dat')]

new_labels_list = [x.replace('.dat','') for x in label_names]
#full_parts_list['Part_num']=full_parts_list.index
#new_labels_list=new_labels_list[1800:1890]
#print(new_labels_list)
#Create List of Parts from non-standard with letters in
for partn in new_labels_list:
    
    partn_m = re.findall('\d+',partn)
    if partn.isnumeric() == True:
        parts_list =parts_list.append(pd.Series([partn, partn], index=parts_list.columns) , ignore_index=True)
       
    else:
        if len(partn_m) > 0:
            if partn_m[0] not in parts_list:
                parts_list =parts_list.append(pd.Series([partn_m[0],partn], index=parts_list.columns) , ignore_index=True)

### Calculate Amount of Variants
Variant_column = []
parts_list['Num_var'] = parts_list.groupby('Directory')['Part_num'].transform('nunique')

#print (parts_list)

#print(parts_list[:200])
print('Part {} Directories {} '.format(parts_list['Part_num'].nunique(),parts_list['Directory'].nunique()))
parts_list.to_csv(dataset_path+part_list_csv, encoding='utf-8', index=True)
##
