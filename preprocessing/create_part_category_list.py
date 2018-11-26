import matplotlib.pyplot as plt

# from preprocessing.create_dataset import create_part_category_list



from preprocessing.create_dataset import generate_single_part_dataset

"""
df = create_part_category_list('res/parts_13463')

print("number of parts in total")
print(len(df.index))

category_counts = df.category.value_counts()
#category_counts = category_counts[category_counts]
category_counts_sliced = category_counts.loc[category_counts > 10]
print(category_counts_sliced)

fig, ax = plt.subplots(figsize=(10,5))
category_counts_sliced.plot.bar()
fig.tight_layout(pad=2)
ax.set_xlabel("category")
ax.set_ylabel("number of parts")
fig.savefig("results/category_counts.eps", format='eps')
"""


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
