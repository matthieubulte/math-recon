import cv2
import numpy as np
import tensorflow as tf

from shapes.image import *
from learning.classifier import *
from learning.dataset import *

def next_model(models_dir):
    from os import listdir
    return models_dir + "model_%s.ckpt" % (max([int(f[6:].split(".")[0]) for f in listdir(models_dir) if f.startswith("model")] + [0]) + 1)

session = tf.InteractiveSession()

classifier = Classifier(session, 36, summary_path="summaries/", regularization_factor=5e-4, learning_rate=1e-4)
dataset = DataSet("datasets/", batch_size=128, test_part=0.2)

classifier.train(dataset, 2000, path=next_model("models/"))
