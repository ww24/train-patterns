# coding: utf-8
import sys
import numpy as np
import tensorflow as tf
import cv2

sys.path.append("./")
import conv

NUM_CLASSES = 2
IMAGE_SIZE = 64
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE


if __name__ == '__main__':
  test_image = []
  for i in range(1, len(sys.argv)):
    img = cv2.imread(sys.argv[i])
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.flatten().astype(np.float32) / 255.0
    test_image.append(img)
  test_image = np.asarray(test_image)

  images_placeholder = tf.placeholder(tf.float32, shape=(None, IMAGE_PIXELS), name="images_placeholder")
  labels_placeholder = tf.placeholder(tf.int32, shape=(None), name="labels_placeholder")
  keep_prob = tf.placeholder(tf.float32, name="keep_prob")

  with tf.name_scope("inference"):
    logits = conv.inference(images_placeholder, keep_prob, image_size=IMAGE_SIZE)

  saver = tf.train.Saver()
  sess = tf.Session()
  init = tf.initialize_all_variables()
  sess.run(init)

  saver.restore(sess, "model.ckpt")

  for i in range(len(test_image)):
    pred = np.argmax(logits.eval(feed_dict={ 
      images_placeholder: [test_image[i]],
      keep_prob: 1.0 })[0])
    print pred
