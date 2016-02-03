#!/usr/bin/env python
# coding: utf-8

import sys
import random
import commands

OTHER_RATE = 1.5

if len(sys.argv) < 2:
  raise NameError("usage samples.py <size>")

target_ids = [
  # "63644686-bef1-40c5-99ee-b43173b7df39",
  # "b7503c47-80f5-4ac9-8f3e-8a12c568ede0",
  # "e726ee46-2fc0-4055-956d-ed7f0be775fb"
  "bdf8a437-703d-4829-9d5e-f67094aec8cb"
]

items = commands.getoutput("ls images").split("\n")
id_sets = set([item.split("_")[0] for item in items])
print "list size:", len(items)
print "ids:", len(id_sets)

my_list = [item for item in items if item.split("_")[0] in target_ids]
other_list = [item for item in items if not (item.split("_")[0] in target_ids)]
print "my list:", len(my_list)
print "other:", len(other_list)

# add label
my_list = map(lambda item: "0 %s"%item, my_list)
other_list = map(lambda item: "1 %s"%item, other_list)

print "------------"

size = int(sys.argv[1])

# 学習用ファイルリスト
train_files = random.sample(my_list, size)
train_files += random.sample(other_list, int(size * OTHER_RATE))

# 試験用ファイルリスト
test_files = [item for item in my_list if not (item in train_files)]
other_list = [item for item in other_list if not (item in train_files)]
test_files += random.sample(other_list, len(test_files))

random.shuffle(train_files)
random.shuffle(test_files)

print "file:", len(train_files) + len(test_files)

f = open("train.txt", "w")
f.writelines([item + "\n" for item in train_files])
print "train files:", len(train_files)
f.close()

f = open("test.txt", "w")
f.writelines([item + "\n" for item in test_files])
print "test files:", len(test_files)
f.close()
