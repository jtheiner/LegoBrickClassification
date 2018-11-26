import matplotlib.pyplot as plt

from preprocessing.create_dataset import create_part_category_list


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
