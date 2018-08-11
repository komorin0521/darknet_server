#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""
Darknet Yolo Python IF based on https://github.com/pjreddie/darknet
I modify a little, adding class and modify some modules.
"""

import argparse
from ctypes import *
import math
import os
import sys

import numpy as np
import cv2
from PIL import Image


def c_array(ctype, values):
    """
    Converting from python list to c array
    """

    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    """
    BOX class defined by yolo src
    """

    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    """
    DETECTION class defined by yolo src
    """

    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    """
    IMAGE class defined by yolo src(or opencv)
    """

    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    """
    METADATA class defined by yolo src
    """

    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

# load yolo modules
lib = CDLL("./libdarknet.so", RTLD_GLOBAL)

lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

class YoloResult(object):
    """
    YoloResult
    """

    def __init__(self, class_index, obj_name, score, boundingbox):
        """
        Initialization method
        """

        self.class_index = class_index
        self.obj_name = obj_name
        self.score = score
        self.x_min = boundingbox[0] - boundingbox[2]/2 -1
        self.y_min = boundingbox[1] - boundingbox[3]/2 -1
        self.width = boundingbox[2]
        self.height = boundingbox[3]

    def get_detect_result(self):
        """
        getting yolo results
        return dict
        """

        resultdict = {'class_index' : self.class_index,
                      'obj_name' : self.obj_name,
                      'score' : self.score,
                      'bounding_box' : {
                          'x_min' : self.x_min,
                          'y_min' : self.y_min,
                          'width' : self.width,
                          'height' : self.height}
                     }
        return resultdict

    def show(self):
        """
        show result
        """

        print('class_index : %d' % self.class_index)
        print('obj_name    : %s' % self.obj_name)
        print('score       : %.2f' % self.score)
        print('bbox.x_min  : %.2f' % self.x_min)
        print('bbox.y_min  : %.2f' % self.y_min)
        print('bbox.width  : %.2f' % self.width)
        print('bbox.height : %.2f' % self.height)

class Yolo(object):
    """
    Yolo class
    """

    def __init__(self, cfgfilepath, datafilepath, weightsfilepath):
        """
        Initialization method
        """

        self.net = load_net(cfgfilepath, weightsfilepath, 0)
        self.meta = load_meta(datafilepath)
        self._colors = [[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0],
                        [1, 1, 0], [1, 0, 0]]

    def _convert_to_yolo_img(self, img):
        """
        converting from cv2 image class to yolo image class
        """

        img = img / 255.0
        h, w, c = img.shape
        img = img.transpose(2, 0, 1)
        outimg = make_image(w, h, c)
        img = img.reshape((w*h*c))
        data = c_array(c_float, img)
        outimg.data = data
        rgbgr_image(outimg)
        return outimg

    def _get_color(self, c, x, max_num):
        """
        Getting color based on yolo src
        """

        ratio = 5*(float(x)/max_num)
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio -= i
        r = (1 - ratio) * self._colors[i][c] + ratio*self._colors[j][c]
        return int(255*r)

    def save_img(self, img, outputfilepath):
        """
        Saving img
        """

        try:
            cv2.imwrite(outputfilepath, img)
        except cv2.error as cv2_err:
            print(cv2_err)
            raise cv2_err

    def draw_detections(self, img, yolo_results):
        """
        drawing result of yolo
        """

        _, height, _ = img.shape
        for yolo_result in yolo_results:
            class_index = yolo_result.class_index
            obj_name = yolo_result.obj_name
            x = yolo_result.x_min
            y = yolo_result.y_min
            w = yolo_result.width
            h = yolo_result.height

            offset = class_index * 123457 % self.meta.classes

            red = self._get_color(2, offset, self.meta.classes)
            green = self._get_color(1, offset, self.meta.classes)
            blue = self._get_color(0, offset, self.meta.classes)
            box_width = int(height * 0.006)
            cv2.rectangle(img, (int(x), int(y)), (int(x+w)+1, int(y+h)+1), (red, green, blue), box_width)
            cv2.putText(img, obj_name, (int(x) -1, int(y) -1), cv2.FONT_HERSHEY_PLAIN, 2, (red, green, blue), 2)

        return img

    def predict(self, img, thresh=.5, hier_thresh=.5, nms=.45):
        """
        Predicting using yolo
        """

        image = self._convert_to_yolo_img(img)
        num = c_int(0)
        pnum = pointer(num)
        predict_image(self.net, image)
        dets = get_network_boxes(self.net, image.w, image.h, thresh, hier_thresh, None, 0, pnum)
        num = pnum[0]
        if (nms): do_nms_obj(dets, num, self.meta.classes, nms);

        res = []
        for j in range(num):
            for i in range(self.meta.classes):
                if dets[j].prob[i] > 0:
                    b = dets[j].bbox
                    res.append(YoloResult(i, self.meta.names[i].decode(), dets[j].prob[i], (b.x, b.y, b.w, b.h)))

        # res = sorted(res, key=lambda x: -x[1])
        # free_image(image)
        free_detections(dets, num)
        return res

def check_exist_path(filepath):
    """
    Checking the filepath
    """

    if not os.path.exists(filepath):
        raise NameError("%s is not found" % filepath)

def importing_args():
    """
    Importing arguments
    """

    parser = argparse.ArgumentParser("Darknet Yolo Python Interfase")
    parser.add_argument("--cfgfilepath", "-cf", required=True)
    parser.add_argument("--datafilepath", "-df", required=True)
    parser.add_argument("--weightfilepath", "-wf", required=True)
    parser.add_argument("--inputfilepath", "-if", required=True)
    parser.add_argument("--outputfilepath", "-of", required=True)
    args = parser.parse_args()

    if args.outputfilepath.find(os.path.sep) < 0:
        cheking_filepathlist = [args.cfgfilepath, args.datafilepath, args.weightfilepath,
                                args.inputfilepath]
    else:
        cheking_filepathlist = [args.cfgfilepath, args.datafilepath,
                                args.weightfilepath, args.inputfilepath,
                                os.path.dirname(args.outputfilepath)]

    for filepath in cheking_filepathlist:
        try:
            check_exist_path(filepath)
        except NameError as name_err:
            print(name_err)
            sys.exit(1)

    return args.cfgfilepath, args.datafilepath, args.weightfilepath, \
            args.inputfilepath, args.outputfilepath

def predict_from_cv2(yolo, inputfilepath):
    """
    Predicting from cv2 format
    yolo: Yolo class
    inputfilepath: filepath of image
    """

    print("call func of predict_from_cv2")
    img = cv2.imread(inputfilepath)
    yolo_results = yolo.predict(img)
    for yolo_result in yolo_results:
        print(yolo_result.get_detect_result())


def predict_from_pil(yolo, inputfilepath):
    """
    Predicting from PIL format
    yolo: Yolo class
    inputfilepath: filepath of image
    """

    print("call func of predict_from_pil")
    img = np.array(Image.open(inputfilepath))
    yolo_results = yolo.predict(img)
    for yolo_result in yolo_results:
        print(yolo_result.get_detect_result())

def main():
    """
    Main
    """
    cfgfilepath, datafilepath, weightfilepath, inputfilepath, outputfilepath = importing_args()
    yolo = Yolo(cfgfilepath.encode(), datafilepath.encode(), weightfilepath.encode())
    print("=====================================")
    predict_from_cv2(yolo, inputfilepath)
    print("=====================================")
    predict_from_pil(yolo, inputfilepath)

if __name__ == "__main__":
    main()
