import cv2
import numpy as np
import tensorflow as tf

class Classifier:
    def __init__(self, session, labels_count, summary_path="/tmp/summaries/", regularization_factor=5e-4, learning_rate=1e-4):
        self.session = session
        self.labels_count = labels_count

        self.__build_network_structure(learning_rate, regularization_factor)
        self.__build_summaries(summary_path)

        self.saver = tf.train.Saver()

        self.session.run(tf.initialize_all_variables())

    def __build_summaries(self, summary_path):
        tf.scalar_summary("accuracy", self.accuracy)
        tf.scalar_summary("loss", self.loss)

        self.summaries = tf.merge_all_summaries()
        self.train_writer = tf.train.SummaryWriter(summary_path + "train", self.session.graph)
        self.test_writer = tf.train.SummaryWriter(summary_path + "test")


    def __build_network_structure(self, learning_rate, regularization_factor):
        self.x = tf.placeholder(tf.float32, shape=[None, 784], name="x")

        # reshape input tensor into an image
        x_im = tf.reshape(self.x, [-1, 28, 28, 1])

        # first conv layer
        W_conv1 = Classifier.weights([5, 5, 1, 32], "W_conv1") # 32 times 5 * 5 patches
        b_conv1 = Classifier.biases([32], "b_conv1")

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

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # add dropout between the densly connected layers
        self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # second densly connected layer
        W_fc2 = Classifier.weights([1024, self.labels_count], "W_fc2")
        b_fc2 = Classifier.biases([self.labels_count], "b_fc2")

        self.y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        # training part
        self.y_ = tf.placeholder(tf.float32, shape=[None, self.labels_count])

        # delta to target label
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y_conv), reduction_indices=[1]))

        # regularize the densly connected weights
        regularizers = regularization_factor * (tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) +
                        tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(b_fc2))

        self.loss = self.cross_entropy + regularizers

        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        correct_prediction = tf.equal(tf.argmax(self.y_conv,1), tf.argmax(self.y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def restore_model_from(self, path):
        self.saver.restore(self.session, path)

    def save_model_to(self, path):
        self.saver.save(self.session, path)

    def __make_feed(self, dataset, test):
        if test:
            return { self.x:dataset.test_images, self.y_: dataset.test_labels, self.keep_prob: 1.0}
        else:
            batch = dataset.next_batch()
            return {self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5}

    def __save_model(self, path):
        if path is not None:
            self.saver.save(self.session, path)

    def train(self, dataset, batches, path=None):
        for i in xrange(batches):
            if i % 10 == 0:
                test_accuracy, summary = self.session.run([self.accuracy, self.summaries], feed_dict=self.__make_feed(dataset, True))
                self.test_writer.add_summary(summary, i)
                self.__save_model(path)

                print "Test accuracy at step %s: %s" % (i, test_accuracy)

            else:
                _, summary = self.session.run([self.train_step, self.summaries], feed_dict=self.__make_feed(dataset, False))
                self.train_writer.add_summary(summary, i)

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
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

    @staticmethod
    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding="SAME")
