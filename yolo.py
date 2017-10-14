#!/usr/env python2
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import


import argparse
import sys
import os

from flask import Flask, request, redirect, jsonify
from werkzeug import secure_filename



sys.path.append(os.path.join(os.getcwd(),'python/'))
import darknet as dn

class Yolo(object):
    def __init__(self, cfgfilepath, weightfilepath, datafilepath, thresh=0.25):
        print(cfgfilepath)
        print(weightfilepath)
        self.net = dn.load_net(cfgfilepath, weightfilepath, 0)
        self.meta = dn.load_meta(datafilepath)
        self.thresh = thresh

    def detect(self, filepath):
        results = dn.detect(self.net, self.meta, filepath, self.thresh)
        return results

def importargs():
    parser = argparse.ArgumentParser('This is the python script of yolo.This is only detect the objects')

    parser.add_argument("--cfgfilepath", "-cf", help = "config filepath  of darknet", type=str)
    parser.add_argument("--datafilepath", "-df", help = "datafilepath of darknet", type=str)
    parser.add_argument("--weightfilepath", "-wf", help = "weight filepath of darknet")
    parser.add_argument("--imagefilepath", "-if", help = "image filepath you want to detect")

    args = parser.parse_args()

    assert os.path.exists(args.cfgfilepath), "cfgfilepath of %s does not exist" % args.cfgfilepath

    assert os.path.exists(args.datafilepath), "datafilepath of %s does not exist" % args.datafilepath

    assert os.path.exists(args.weightfilepath), "weightfilepath of %s does not exist" % args.weightfilepath

    assert os.path.exists(args.weightfilepath), "imagefilepath of %s does not exist" % args.imagefilepath

    args = parser.parse_args()

    return args.cfgfilepath, args.datafilepath, args.weightfilepath, args.imagefilepath

def main():
    cfgfilepath, datafilepath, weightfilepath, imagefilepath = importargs()

    yolo = Yolo(cfgfilepath, weightfilepath, datafilepath)
    yolo_results = yolo.detect(imagefilepath)

    for yolo_result in yolo_results:
        print("obj = %s, score = %.3f bounding_box = [ %f, %f, %f, %f]" % (yolo_result[0], yolo_result[1], yolo_result[2][0], yolo_result[2][1], yolo_result[2][2], yolo_result[2][3]))

if __name__ == "__main__":
    main()
