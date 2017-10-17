#!/usr/env python2
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import


import argparse
import sys
import os

import cv2

sys.path.append(os.path.join(os.getcwd(),'python/'))
import darknet as dn

class YoloResult(object):
    def __init__(self, obj_name, score, boundingbox):
        self.obj_name = obj_name
        self.score = score
        self.x_central = boundingbox[0]
        self.y_central = boundingbox[1]
        self.width = boundingbox[2]
        self.height = boundingbox[3]

    def get_detect_result(self):
        resultdict = { 'obj_name' : self.obj_name,
                       'score' : self.score,
                       'bounding_box' : {
                           'x_central' : self.x_central,
                           'y_central' : self.y_central,
                           'width' : self.width,
                           'height' : self.height }
                       }
        return resultdict


class Yolo(object):
    def __init__(self, cfgfilepath, weightfilepath, datafilepath, thresh=0.25):
        print(cfgfilepath)
        print(weightfilepath)
        self.net = dn.load_net(cfgfilepath, weightfilepath, 0)
        self.meta = dn.load_meta(datafilepath)
        self.thresh = thresh

    def detect(self, filepath):
        raw_results = dn.detect(self.net, self.meta, filepath, self.thresh)

        detect_results = list()
        for raw_result in raw_results:
            yolo_result = YoloResult(raw_result[0], raw_result[1], raw_result[2])
            print(yolo_result.get_detect_result())
            detect_results.append(yolo_result)
        return detect_results

    def insert_rectangle(self, filepath, yolo_results, outputdir='outputdir'):
        img = cv2.imread(filepath, 1)
        for yolo_result in yolo_results:
            obj_name = yolo_result.obj_name
            x = yolo_result.x_central - yolo_result.width/2 -1
            y = yolo_result.y_central - yolo_result.height/2 -1
            w = yolo_result.width
            h = yolo_result.height

            cv2.rectangle(img, (int(x), int(y)), (int(x+w+1), int(y+h+1)), 3)
            cv2.putText(img, obj_name, (int(x) -1, int(y) -1), cv2.FONT_HERSHEY_PLAIN, 2, 3)
        outputfilename = filepath.split(os.path.sep)[-1]

        outputfilepath = os.path.join(outputdir, outputfilename)
        cv2.imwrite(outputfilepath, img)

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
    yolo.insert_rectangle(imagefilepath, yolo_results)
    # for yolo_result in yolo_results:
        # print("obj = %s, score = %.3f bounding_box = [ %f, %f, %f, %f]" % (yolo_result[0], yolo_result[1], yolo_result[2][0], yolo_result[2][1], yolo_result[2][2], yolo_result[2][3]))

if __name__ == "__main__":
    main()
