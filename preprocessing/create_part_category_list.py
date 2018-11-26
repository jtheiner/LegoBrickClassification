import matplotlib.pyplot as plt

from preprocessing.create_dataset import generate_single_part_dataset

# todo: !

import os
import pandas as pd
import numpy as np

with open("res/top100.csv", 'r') as csv:
    part_numbers = []
    for i, line in enumerate(csv):
        id = line.split(" ")[1]
        id = id.replace("\t", "")

        part_numbers.append(id)

    with open('results/top100_parts.csv', 'w') as f:
        for item in part_numbers:
            f.write("%s\n" % item)


#df = pd.read_csv("results/top100_parts.csv",sep='\t')


dat_directory = "res/parts_13463"

#parts = list(map(str, df.ix[:, 0].values.tolist()))



parts = ["2357", "2412", "2420", "2431", "2540", "2654", "2780", "2877", "3001" , "3002"]
files = []
for file in os.listdir(dat_directory):
    if file.endswith(".dat") and file[:-4] in parts:
        files.append(file)

print(files)
print(len(files))

for file in files:
    generate_single_part_dataset(os.path.join(dat_directory, file),
                                 "res/bg_noise/",
                                 "results/dataset_top10/", 200)
