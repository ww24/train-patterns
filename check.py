#!/usr/bin/env python
# coding: utf-8

import subprocess as sp

cmd = "sh -c '%s'"%'python test.py $(cat test.txt | grep "0 " | awk \'"\'"\'{print "images/"$2;}\'"\'"\')'

mohemohe_res = sp.check_output(cmd, shell=True)
others_res = sp.check_output("1 ".join(cmd.split("0 ")), shell=True)

mohemohe = [line.strip() for line in mohemohe_res.strip().split("\n")]
others = [line.strip() for line in others_res.strip().split("\n")]

mohemohe_all = len(mohemohe)
others_all = len(others)

mohemohe_correct = len([line for line in mohemohe if line == "0"])
others_correct = len([line for line in others if line == "1"])

print "mohe_all: %d"%mohemohe_all
print "others_all: %d"%others_all

print "mohe_correct: %d"%mohemohe_correct
print "others_correct: %d"%others_correct

print "mohe rate: %g"%(mohemohe_correct / float(mohemohe_all))
print "others rate: %g"%(others_correct / float(others_all))
