# coding: utf-8
#
# Simple, end-to-end, LeNet-5-like convolutional MNIST model example.
# ref: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/image/mnist/convolutional.py

import re
import sys
import cv2
import math
import random
import numpy as np
import tensorflow as tf
import tensorflow.python.platform

# 予測モデルの作成
def inference(images, keep_prob, **kwargs):
  if "IMAGE_SIZE" in globals():
    image_size = kwargs.get("image_size", IMAGE_SIZE)
  else:
    image_size = kwargs.get("image_size", 32)

  if "NUM_CLASSES" in globals():
    num_classes = kwargs.get("num_classes", NUM_CLASSES)
  else:
    num_classes = kwargs.get("num_classes", 2)

  def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

  def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

  x_image = tf.reshape(images, [-1, image_size, image_size, 1])

  # 畳み込み1
  with tf.name_scope("conv1"):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # プーリング1
  with tf.name_scope("pool1"):
    h_pool1 = max_pool_2x2(h_conv1)

  # 畳み込み2
  with tf.name_scope("conv2"):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # プーリング2
  with tf.name_scope("pool2"):
    h_pool2 = max_pool_2x2(h_conv2)

  # 全結合1
  with tf.name_scope("fc1"):
    # size = 7 * 7 * 64
    size = image_size**2 * 4
    W_fc1 = weight_variable([size, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, size])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) * b_fc1)

  # 全結合2
  with tf.name_scope("fc2"):
    W_fc2 = weight_variable([1024, num_classes])
    b_fc2 = bias_variable([num_classes])

  with tf.name_scope("softmax"):
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

  return y_conv

# 損失計算
def loss(logits, labels):
  """Calculates the loss from the logits and the labels.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].
  Returns:
    loss: Loss tensor of type float.
  """
  # Convert from sparse integer labels in the range [0, NUM_CLASSES)
  # to 1-hot dense float vectors (that is we will have batch_size vectors,
  # each with NUM_CLASSES values, all of which are 0.0 except there will
  # be a 1.0 in the entry corresponding to the label).
  batch_size = tf.size(labels)
  labels = tf.expand_dims(labels, 1)
  indices = tf.expand_dims(tf.range(0, batch_size), 1)
  concated = tf.concat(1, [indices, labels])
  onehot_labels = tf.sparse_to_dense(
      concated, tf.pack([batch_size, NUM_CLASSES]), 1.0, 0.0)
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits,
                                                          onehot_labels,
                                                          name='xentropy')
  loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
  tf.scalar_summary("xentropy_mean", loss)
  return loss

# 学習関数
def training(loss, learning_rate):
  """Sets up the training Ops.
  Creates a summarizer to track the loss over time in TensorBoard.
  Creates an optimizer and applies the gradients to all trainable variables.
  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.
  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.
  Returns:
    train_op: The Op for training.
  """
  # tf.scalar_summary(loss.op.name, loss)
  # Create the gradient descent optimizer with the given learning rate.
  # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  optimizer = tf.train.AdamOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op

def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).
  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label's is was in the top k (here k=1)
  # of all logits for that example.
  correct = tf.nn.in_top_k(logits, labels, 1)
  # Return the number of true entries.
  accuracy = tf.reduce_sum(tf.cast(correct, tf.int32))
  accuracy = tf.cast(accuracy, tf.float32) / tf.cast(tf.size(labels), tf.float32, name="accuracy")
  tf.scalar_summary("accuracy", accuracy)
  return accuracy

def activation_summary(x):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def open_list(name):
  f = open(name, "r")
  file_list = [line.strip() for line in f]
  f.close()

  image_label = []
  for line in file_list:
    label, item = line.split(" ")
    filename = "images_%d/%s"%(IMAGE_SIZE, item)
    # グレースケールで画像読み込み
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    # IMAGE_SIZExIMAGE_SIZE に縮小
    # img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.flatten().astype(np.float32) / 255.0
    image_label.append([img, int(label)])
  # シャッフル
  random.shuffle(image_label)
  images = np.asarray([i[0] for i in image_label])
  labels = np.asarray([i[1] for i in image_label])
  print "image_size:", len(images)
  return (images, labels)

def main(_):
  train_images, train_labels = open_list(FLAGS.train)
  test_images, test_labels = open_list(FLAGS.test)

  with tf.Graph().as_default():
    images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, IMAGE_PIXELS), name="images_placeholder")
    labels_placeholder = tf.placeholder(tf.int32, shape=(FLAGS.batch_size), name="labels_placeholder")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    with tf.name_scope("inference"):
      logits = inference(images_placeholder, keep_prob)
    with tf.name_scope("loss"):
      loss_value = loss(logits, labels_placeholder)
    with tf.name_scope("training"):
      train_op = training(loss_value, FLAGS.learning_rate)
    with tf.name_scope("evaluation"):
      acc = evaluation(logits, labels_placeholder)

    saver = tf.train.Saver()
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)

    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph_def=sess.graph_def)

    def calculate_accuracy(type_str, step):
      images = []
      labels = []
      
      if type_str == "train":
        images = train_images
        labels = train_labels
      else:
        images = test_images
        labels = test_labels

      accuracy_mean = 0.0
      summary_all = {}
      for i in xrange(len(images) / FLAGS.batch_size):
        batch = FLAGS.batch_size * i
        accuracy, summary_str = sess.run([acc, summary_op], feed_dict={
          images_placeholder: images[batch:batch + FLAGS.batch_size],
          labels_placeholder: labels[batch:batch + FLAGS.batch_size],
          keep_prob: 1.0})
        accuracy_mean += accuracy
      
        summary = tf.Summary()
        summary.ParseFromString(summary_str)
        for key, val in enumerate(summary.value):
          val.tag = type_str + "/" + val.tag
          if val.tag in summary_all:
            summary_all[val.tag] += val.simple_value
          else:
            summary_all[val.tag] = val.simple_value

      loop_size = math.ceil(len(images) / FLAGS.batch_size)

      summary = tf.Summary()
      for s in summary_all:
        summary.value.add(tag=s, simple_value=summary_all[s] / loop_size)
      summary_writer.add_summary(summary, global_step=step)
      
      accuracy_mean /= loop_size
      print "step %d, %s accuracy %g"%(step, type_str, accuracy_mean)

    # And then after everything is built, start the training loop.
    for step in range(FLAGS.max_steps):
      for i in xrange(len(train_images) / FLAGS.batch_size):
        batch = FLAGS.batch_size * i
        sess.run(train_op, feed_dict={
          images_placeholder: train_images[batch:batch + FLAGS.batch_size],
          labels_placeholder: train_labels[batch:batch + FLAGS.batch_size],
          keep_prob: 0.5})

      with tf.name_scope("train") as scope:
        # if step % 100 == 0:
        calculate_accuracy(scope[:-1].split("_")[0], step)

      with tf.name_scope("test") as scope:
        # if step % 100 == 0:
        calculate_accuracy(scope[:-1].split("_")[0], step)

      summary_writer.flush()

  save_path = saver.save(sess, "model.ckpt")


if __name__ == "__main__":
  flags = tf.app.flags
  FLAGS = flags.FLAGS
  flags.DEFINE_string("train", "train.txt", "File path of train image list")
  flags.DEFINE_string("test", "test.txt", "File path of test image list")
  flags.DEFINE_float("learning_rate", 1e-5, "Initial learning rate.")
  flags.DEFINE_integer("max_steps", 1000, "Number of steps to run trainer.")
  flags.DEFINE_integer("batch_size", 10, "Batch size."
                       "Must divide evenly into the dataset sizes.")
  flags.DEFINE_string("train_dir", "data", "Directory to put the training data.")
  flags.DEFINE_integer("image_size", 64, "Image size.")

  NUM_CLASSES = 2
  TOWER_NAME = "tower"
  IMAGE_SIZE = FLAGS.image_size
  IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

  tf.app.run()
