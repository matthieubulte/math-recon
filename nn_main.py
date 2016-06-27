import cv2
import numpy as np
import tensorflow as tf

from shapes.image import *
from classifier import Classifier

session = tf.InteractiveSession()

# labels are: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
classifier = Classifier(session, 10)

# read dataset
images = []
labels = []
for i in range(1, 11):
    for j in range(1, 56):
        f = "dataset/img%03d-%03d.png" % (i, j)

        label = np.zeros(10)
        label[i - 1] = 1

        ims, lbls = Image.from_file(f).generate_training_items(label)
        images += ims
        labels += lbls

images = np.array(images)
labels = np.array(labels)

# dataset utils
dataset_size = len(images)
dataset_index = 0

perm = np.arange(dataset_size)
np.random.shuffle(perm)
images = images[perm]
labels = labels[perm]

def next_batch(size):
    global dataset_index
    global dataset_size
    global images
    global labels
    global mnist

    if dataset_index + size > dataset_size:
        dataset_index = 0

        perm = np.arange(dataset_size)
        np.random.shuffle(perm)
        images = images[perm]
        labels = labels[perm]

    start = dataset_index
    dataset_index += size

    return images[start:dataset_index], labels[start:dataset_index]

def next_model(models_dir):
    from os import listdir
    return models_dir + "model_%s.ckpt" % (max([int(f[6:].split(".")[0]) for f in listdir(models_dir) if f.startswith("model")]) + 1)

classifier.train(lambda: next_batch(64), 20000, path=next_model("models/"))
