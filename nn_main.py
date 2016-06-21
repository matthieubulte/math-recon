import cv2
import numpy as np
import tensorflow as tf

from classifier import Classifier

session = tf.InteractiveSession()

classifier = Classifier(session)
classifier.restore_model_from("model.ckpt")

for i in range(0, 16):
    path = "images/symbols/sym_%s.png" % i
    image = cv2.imread(path)

    print("%s: %s" % (path, classifier.classify(image)))


return


# TODO put the training part in the classifier class

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
y_ = tf.placeholder(tf.float32, shape=[None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

saver = tf.train.Saver()

for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))

    save_path = saver.save(sess, "model.ckpt")
    print("saved")
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
