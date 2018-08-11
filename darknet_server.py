#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import datetime
import io
import sys
import os

import cv2
from flask import Flask, request, redirect, jsonify
from flask import send_file
from werkzeug import secure_filename
from pykakasi import kakasi


from darknet import Yolo
from darknet import YoloResult
# from read_conf import read_conf

class DarknetServer(Flask):
    def __init__(self, host, name, upload_dir, extensions, pub_img_flag, yolo):
        """
        init server class
        """
        super(DarknetServer, self).__init__(name)
        self.host = host
        self.config['UPLOAD_FOLDER'] = upload_dir
        self.extensions = extensions
        self.yolo = yolo
        self.converter = None
        self.pub_img_flag = pub_img_flag
        self.define_uri()

    def define_uri(self):
        """
        definition of uri
        """
        self.provide_automatic_option = False
        self.add_url_rule('/detect', None, self.detect, methods = [ 'POST' ] )
        self.add_url_rule('/get_predict_image', None, self.get_predict_image, methods = [ 'POST' ] )

    def setup_converter(self):
        """
        """
        mykakasi = kakasi()
        mykakasi.setMode('H', 'a')
        mykakasi.setMode('K', 'a')
        mykakasi.setMode('J', 'a')
        self.converter = mykakasi.getConverter()

    def convert_filename(self, filename):
        """
        converting filename using pykakasi
        """
        return self.converter.do(filename)

    def check_allowfile(self, filename):
        """
        checking extenson
        """
        if len(filename.split(".")) > 1:
            extension = filename.split(".")[-1]
            print("extension is %s" % extension)
            return extension in self.extensions
        else:
            return False

    def get_yolo_results(self, request):
        """
        Getting yolo results
        @param: request
        @return: the list of yolo result
        """
        file = request.files['file']
        if file and self.check_allowfile(file.filename):
            print("receive the file, the filename is %s" % file.filename)
            output_filename = "%s_%s" % (datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), self.convert_filename(file.filename))
            print("output filename is %s" % output_filename)
            outputfilepath = os.path.join(self.config['UPLOAD_FOLDER'], output_filename)
            file.save(outputfilepath)

            try:
                img = cv2.imread(outputfilepath)
            except Exception as err:
                print(err)

            if request.form.get("thresh"):
                thresh = float(request.form.get("thresh"))
                print("the request parameter of thresh hold is %f" % thresh)
                yolo_results = self.yolo.predict(img, thresh)
            else:
                print("the threshold is not included of parameter")
                yolo_results = self.yolo.predict(img)
                for yolo_result in yolo_results:
                    print("=========================")
                    yolo_result.show()

            return img, yolo_results, outputfilepath

    def detect(self):
        """
        Detection using yolo. '/detect'
        """
        print("call api of detect")
        if request.method == 'POST':
            img, yolo_results, outputfilepath = self.get_yolo_results(request)
            res = dict()
            res['status'] = '200'
            res['result'] = list()
            for yolo_result in yolo_results:
                res['result'].append(yolo_result.get_detect_result())

            if self.pub_img_flag:
                try:
                    pred_img = self.yolo.draw_detections(img, yolo_results)
                    tmpfilename = outputfilepath.split(os.path.sep)[-1]
                    outputfilename = "%s_pred.jpg" % tmpfilename.split('.')[0]

                    self.yolo.save_img(pred_img, '/var/www/html/images/%s' % outputfilename)
                    filename = outputfilepath.split(os.path.sep)[-1]
                    res['image_src'] = 'http://%s/images/%s' % (self.host, outputfilename)

                except Exception as e:
                    print("An error occured")
                    print("The information of error is as following")
                    print(type(e))
                    print(e.args)
                    print(e)

            return jsonify(res)
        else:
            res = dict()
            res['status'] = '500'
            res['msg'] = 'The file format is only jpg or png'

    def get_predict_image(self):
        """
        Getting yolo result
        """
        print("call api of get_predict_image")
        if request.method == 'POST':
            print("get yolo results")
            img, yolo_results, outputfilepath = self.get_yolo_results(request)

            print("draw detections")
            pred_img = self.yolo.draw_detections(img, yolo_results)

            tmpfilename = outputfilepath.split(os.path.sep)[-1]
            print(tmpfilename)
            pred_outputfilename = "%s_pred.jpg" % tmpfilename.split('.')[0]
            pred_img_outputfilepath = os.path.join(self.config['UPLOAD_FOLDER'], pred_outputfilename)

            print("pred img outputfilepath: %s" % pred_img_outputfilepath)
            self.yolo.save_img(pred_img, pred_img_outputfilepath)

            with open(pred_img_outputfilepath, 'rb') as img:
                return send_file(io.BytesIO(img.read()),
                        attachment_filename=pred_outputfilename,
                        mimetype='image/%s' % pred_outputfilename.split('.')[-1])

        else:
            res = dict()
            res['status'] = '500'
            res['msg'] = 'The file format is only jpg or png'

def importargs():
    parser = argparse.ArgumentParser('This is a server of darknet')

    parser.add_argument("--cfgfilepath", "-cf", help = "config filepath  of darknet", type=str, required=True)
    parser.add_argument("--datafilepath", "-df", help = "datafilepath of darknet", type=str, required=True)
    parser.add_argument("--weightfilepath", "-wf", help = "weight filepath of darknet", required=True)
    parser.add_argument("--host", "-H", help = "host name running server",type=str, required=False, default='localhost')
    parser.add_argument("--port", "-P", help = "port of runnning server", type=int, required=False, default=8080)
    parser.add_argument("--uploaddir", "-ud", help = "upload folder of images")
    parser.add_argument("--publish-image-flag","-pf", help="If true, outputting the image of /var/www/html and add the image src to response",type=str, required=False, default="False", choices= [ "true", "false", "True", "False" ])

    args = parser.parse_args()

    assert os.path.exists(args.cfgfilepath), "cfgfilepath of %s does not exist" % args.cfgfilepath
    assert os.path.exists(args.datafilepath), "datafilepath of %s does not exist" % args.datafilepath
    assert os.path.exists(args.weightfilepath), "weightfilepath of %s does not exist" % args.weightfilepath
    assert os.path.exists(args.uploaddir) & os.path.isdir(args.uploaddir), "uploaddir of %s does not exist or is not directory" % args.uploaddir

    if args.publish_image_flag in [ "True", "true" ]:
        publish_image_flag = True
    elif args.publish_image_flag in [ "False", "false" ]:
        publish_image_flag = False

    return args.cfgfilepath, args.datafilepath, args.weightfilepath, args.host, args.port, args.uploaddir, publish_image_flag


def main():
    cfgfilepath, datafilepath, weightfilepath, host, port, uploaddir, pub_img_flag = importargs()
    yolo = Yolo(cfgfilepath.encode(), datafilepath.encode(), weightfilepath.encode())

    server = DarknetServer(host, 'yolo_server', uploaddir, [ 'jpg', 'png' ], pub_img_flag, yolo )
    server.setup_converter()
    print("server run")
    server.run(host=host, port=port)



if __name__ == "__main__":
    main()
