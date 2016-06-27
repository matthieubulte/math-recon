%matplotlib inline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# read dataset
labels = []
for i in range(1, 63):
    upper = 56 if i <= 10 else 2
    for j in range(1, upper):
        label = np.zeros(11)

        if i <= 10:
            # digit label speicific to digit
            label[i - 1] = 1
        else:
            # non digit label grouping all non digits
            label[-1] = 1

        labels.append(label)
labels = np.array(labels)

dataset_size = len(labels)
dataset_index = 0

perm = np.arange(dataset_size)
np.random.shuffle(perm)
labels = labels[perm]

def next_batch(size):
    global dataset_index
    global dataset_size
    global labels

    if dataset_index + size > dataset_size:
        dataset_index = 0

        perm = np.arange(dataset_size)
        np.random.shuffle(perm)
        labels = labels[perm]

    start = dataset_index
    dataset_index += size

    return labels[start:dataset_index]

result = np.zeros(11)

for i in range(20000):
    result += next_batch(50).sum(axis=0)

df = pd.DataFrame({ 'x': range(1, 12), 'y': result })
result
df.plot()

np.var(result)
np.average(result)
np.std(result)
