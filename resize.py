#!/usr/bin/env python
# coding: utf-8

import sys
import os
import cv2

if len(sys.argv) < 2:
  raise NameError("usage resize.py <size>")
size = int(sys.argv[1])

files = os.listdir("images/")
images = [file for file in files if file.rsplit(".")[-1:][0] == "png"]
print len(images)

directory = "images_%d"%size

try:
  os.mkdir(directory)
except OSError:
  print directory + " exists."

for image in images:
  print "file:%s"%image
  img = cv2.imread("images/"+image, cv2.IMREAD_GRAYSCALE)
  img = cv2.resize(img, (size, size))
  cv2.imwrite(directory + "/" + image, img)
