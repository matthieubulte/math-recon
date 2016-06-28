import cv2
import numpy as np
import tensorflow as tf

from shapes.image import *
from learning.classifier import *
from learning.dataset import *

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

session = tf.InteractiveSession()

classifier = Classifier(session, 10)
classifier.restore_model_from("models/model_1.ckpt")

print classifier.evaluate_accuracy((mnist.test.images, mnist.test.labels))
