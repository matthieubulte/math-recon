import cv2
import numpy as np
import tensorflow as tf

from shapes.image import *
from classifier import Classifier
from dataset import DataSet

def next_model(models_dir):
    from os import listdir
    return models_dir + "model_%s.ckpt" % (max([int(f[6:].split(".")[0]) for f in listdir(models_dir) if f.startswith("model")] + [0]) + 1)

session = tf.InteractiveSession()

classifier = Classifier(session, 10)
dataset = DataSet("dataset_2/", batch_size=32, test_part=0.2)

classifier.train(dataset, 500, path=next_model("models/"))
