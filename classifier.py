import cv2
import numpy as np
import tensorflow as tf

class Classifier:
    def __init__(self, session, labels_count):
        self.session = session
        self.labels_count = labels_count
        self.__build_network_structure()

    def __build_network_structure(self):
        self.x = tf.placeholder(tf.float32, shape=[None, 784], name="x")

        # first conv layer
        W_conv1 = Classifier.weights([5, 5, 1, 32], "W_conv1") # 32 times 5 * 5 patches
        b_conv1 = Classifier.biases([32], "b_conv1")

        x_im = tf.reshape(self.x, [-1, 28, 28, 1])

        h_conv1 = tf.nn.relu(Classifier.conv2d(x_im, W_conv1) + b_conv1)
        h_pool1 = Classifier.max_pool_2x2(h_conv1)

        # second conv layer
        W_conv2 = Classifier.weights([5, 5, 32, 64], "W_conv2")
        b_conv2 = Classifier.biases([64], "b_conv2")

        h_conv2 = tf.nn.relu(Classifier.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = Classifier.max_pool_2x2(h_conv2)

        # first densly connected layer
        W_fc1 = Classifier.weights([7 * 7 * 64, 1024], "W_fc1")
        b_fc1 = Classifier.biases([1024], "b_fc1")

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # second densly connected layer
        W_fc2 = Classifier.weights([1024, self.labels_count], "W_fc2")
        b_fc2 = Classifier.biases([self.labels_count], "b_fc2")

        self.y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        # training part
        self.y_ = tf.placeholder(tf.float32, shape=[None, self.labels_count])

        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y_conv), reduction_indices=[1]))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(self.y_conv,1), tf.argmax(self.y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.session.run(tf.initialize_all_variables())

    def restore_model_from(self, path):
        saver = tf.train.Saver()
        saver.restore(self.session, path)

    def save_model_to(self, path):
        saver = tf.train.Saver()
        saver.save(self.session, path)

    def train(self, next_batch, batches, path=None):
        if path is not None:
            saver = tf.train.Saver()

        for i in range(batches):
          batch = next_batch()
          if i%100 == 0:
            train_accuracy = self.accuracy.eval(feed_dict={
                self.x:batch[0], self.y_: batch[1], self.keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))

            if path is not None:
                saver.save(self.session, path)

          self.train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})

    def evaluate_accuracy(self, batch):
        return self.accuracy.eval(feed_dict={self.x:batch[0], self.y_: batch[1], self.keep_prob: 1.0})

    def classify(self, image):
        prediction = tf.argmax(self.y_conv, 1)
        result = self.session.run(prediction, feed_dict={self.x : [image], self.keep_prob: 1.0})

        return result[0]

    @staticmethod
    def weights(shape, name):
        W = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(W, name=name)

    @staticmethod
    def biases(shape, name):
        b = tf.constant(0.1, shape=shape)
        return tf.Variable(b, name=name)

    @staticmethod
    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')
