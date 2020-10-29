import cv2
import imutils
import numpy as np
import fnmatch
import os
import logging
import pandas as pd
import re



dataset_path = '/Users/petesmac/Documents/Machine Learning/DATA/LEGO-brick-images/'
Ldraw_path = '/Users/petesmac/Documents/Machine Learning/LegoBrickClassification/'

full_parts_list = pd.read_csv(dataset_path+'full_parts.csv', encoding='utf-8', index_col='part_num')
col_names =  ['Part_num']
parts_list  = pd.DataFrame(columns = col_names)
col_names =  ['Directory', 'Part_num']
Var_list = pd.DataFrame(columns = col_names ) #creates a new dataframe that's empty
 
#print (len(dataset_files))
#count=0
##for partn in dataset_files:
##    
##    if partn.isnumeric() == True:
##        count=count+1
##print(count)

label_names = [str(filenames) for  filenames in fnmatch.filter(os.listdir(Ldraw_path+'ldraw/parts'), '*.dat')]   
new_labels_list = [x.replace('.dat','') for x in label_names]
full_parts_list['part_num']=full_parts_list.index
#Create List of Parts from non-standard with letters in
for partn in new_labels_list:
    
    partn_m = re.findall('\d+',partn)
    if partn.isnumeric() == True:
        parts_list =parts_list.append(pd.Series([partn], index=parts_list.columns) , ignore_index=True)
        Var_list =Var_list.append(pd.Series([partn,partn], index=Var_list.columns) , ignore_index=True)
    else:
        if len(partn_m) > 0:
            if partn_m[0] not in parts_list:
                parts_list =parts_list.append(pd.Series([partn_m[0]], index=parts_list.columns) , ignore_index=True)
            Var_list =Var_list.append(pd.Series([partn_m[0],partn], index=Var_list.columns) , ignore_index=True) 
print("Number Brick",len(parts_list))


###Add Description
##Description_list=[]
##for partn in parts_list['Part_num']:
##    
##    P=0
##    Des=""
##    for f_partn in full_parts_list['part_num']:
##            if f_partn == partn:
##                Des = full_parts_list['name'].iloc[P]
##                #print(Des)
##            Description_list.append(Des)
##                
##            P=P+1
##    #print(Description_list)
##parts_list.insert(1, "Description",Description_list , True)
                
#Create Variant
Variant_column = []
#for part in parts_list['Part_num']:
#    partn_m = re.findall('\d+',part)
#    print(part)
    #a=Var_list["Directory"].value_counts()
    #a=Var_list['Directory'].count(part)
    #print(a)
    #Variant_list=[]
    #for v_partn in Var_list:
     #   partn_m = re.findall('\d+',v_partn)
      #  print(partn_m)
        #if len(partn_m) > 1:
#print(Var_list)           
#a=Var_list["Directory"].value_counts()
#print(a)
    #parts_list =parts_list.append(pd.Series([partn,"f"], index=parts_list.columns) , ignore_index=True) 
    #Variant_column.append(Variant_list)
##parts_list.insert(2, "Variant",Variant_column , True)
##print(len(parts_list))
#Var_list.insert(2,'Count', [Var_list["Directory"].value_counts()],True)
parts_list.to_csv(dataset_path+'Parts.csv', encoding='utf-8', index=True)
Var_list.to_csv(dataset_path+'Parts_Alt.csv', encoding='utf-8', index=True)
##
