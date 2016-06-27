%matplotlib inline

import cv2
from classifier import Classifier
import tensorflow as tf
from shapes.image import *

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

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

    if dataset_index + size / 2 > dataset_size:
        dataset_index = 0

        perm = np.arange(dataset_size)
        np.random.shuffle(perm)
        images = images[perm]
        labels = labels[perm]

    start = dataset_index
    dataset_index += size / 2

    mnist_images, mnist_labels = mnist.train.next_batch(size / 2)

    return np.concatenate((images[start:dataset_index], mnist_images), axis=0), np.concatenate((labels[start:dataset_index], mnist_labels), axis=0)

session = tf.InteractiveSession()
classifier = Classifier(session, 10)
classifier.restore_model_from("models/model_6.ckpt")

print classifier.evaluate_accuracy(next_batch(200))
