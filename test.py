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
  test_images = []
  for i in range(1, len(sys.argv)):
    img = cv2.imread(sys.argv[i], cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.flatten().astype(np.float32) / 255.0
    test_images.append(img)
  test_images = np.asarray(test_images)

  images_placeholder = tf.placeholder(tf.float32, shape=(None, IMAGE_PIXELS), name="images_placeholder")
  labels_placeholder = tf.placeholder(tf.int32, shape=(None), name="labels_placeholder")
  keep_prob = tf.placeholder(tf.float32, name="keep_prob")

  with tf.name_scope("inference"):
    logits = conv.inference(images_placeholder, keep_prob, image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)

  saver = tf.train.Saver()
  sess = tf.Session()
  init = tf.initialize_all_variables()
  sess.run(init)

  saver.restore(sess, "models/mohemohe.ckpt")
  # ckpt = tf.train.get_checkpoint_state("./models")
  # if ckpt and ckpt.model_checkpoint_path:
  #   saver.restore(sess, ckpt.model_checkpoint_path)
  # else:
  #   exit("model not found")

  for i in range(len(test_images)):
    res = sess.run(logits, feed_dict={ 
      images_placeholder: [test_images[i]],
      keep_prob: 1.0})
    print "本人: %.4f%%, 他人: %.4f%%"%(res[0][0] * 100, res[0][1] * 100)
    print res[0]
    print np.argmax(res[0])
