#!/usr/bin/env python

import sys
import random
import commands

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

other_list = random.sample(other_list, len(my_list))
file_list = my_list + other_list
random.shuffle(file_list)
print "file list:", len(file_list)

size = int(sys.argv[1])
train_files = random.sample(file_list, size)

f = open("train.txt", "w")
f.writelines([item + "\n" for item in train_files])
print "train files:", len(train_files)
f.close()

f = open("test.txt", "w")
test_files = [item + "\n" for item in file_list if not (item in train_files)]
f.writelines(test_files)
print "test files:", len(test_files)
f.close()
