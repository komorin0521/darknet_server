#!/usr/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import, unicode_literals

import argparse
import datetime
import io
import sys
import os

from flask import Flask, request, redirect, jsonify
from flask import send_file
from werkzeug import secure_filename

sys.path.append(os.path.join(os.getcwd(),'python/'))
import darknet as dn
from yolo import Yolo
from yolo import YoloResult
from pykakasi import kakasi

class MyServer(object):
    def __init__(self, name, host, port, upload_dir, extensions, yolo):
        self.app = Flask(name)
        self.host = host
        self.port = port
        self.app.config['UPLOAD_FOLDER'] = upload_dir
        self.extensions = extensions
        self.yolo = yolo
        self.converter = None


    def setup_converter(self):
        mykakasi = kakasi()
        mykakasi.setMode('H', 'a')
        mykakasi.setMode('K', 'a')
        mykakasi.setMode('J', 'a')
        self.converter = mykakasi.getConverter()


    def convert_filename(self, filename):
        return self.converter.do(filename)

    def check_allowfile(self, filename):
        if len(filename.split(".")) > 1:
            extension = filename.split(".")[-1]
            print("extension is %s" % extension)
            return extension in self.extensions
        else:
            return False


    def get_yolo_results(self, request):
        file = request.files['file']
        if file and self.check_allowfile(file.filename):
            print("receive the file, the filename is %s" % file.filename)
            output_filename = "%s_%s" % (datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), self.convert_filename(file.filename))
            print("output filename is %s" % output_filename)
            outputfilepath = os.path.join(self.app.config['UPLOAD_FOLDER'], output_filename)
            file.save(outputfilepath)
            if request.form.get("thresh"):
                thresh = float(request.form.get("thresh"))
                print("the request parameter of thresh hold is %f" % thresh)
                yolo_results = self.yolo.detect(outputfilepath, thresh)
            else:
                print("the threshold is not included of parameter")
                yolo_results = self.yolo.detect(outputfilepath)
            return yolo_results, outputfilepath


    def detect(self):
        print("call api of detect")
        if request.method == 'POST':
            yolo_results, outputfilepath = self.get_yolo_results(request)
            res = dict()
            res['status'] = '200'
            res['result'] = list()
            for yolo_result in yolo_results:
                res['result'].append(yolo_result.get_detect_result())

            return jsonify(res)
        else:
            res = dict()
            res['status'] = '500'
            res['msg'] = 'The file format is only jpg or png'

    def get_predict_image(self):
        print("call api of get_predict_image")
        if request.method == 'POST':
            yolo_results, outputfilepath = self.get_yolo_results(request)
            predicting_imgfilepath = self.yolo.insert_rectangle(outputfilepath, yolo_results)

            with open(predicting_imgfilepath, 'rb') as img:
                return send_file(io.BytesIO(img.read()),
                        attachment_filename=predicting_imgfilepath.split(os.path.sep)[-1],
                        mimetype='image/%s' % predicting_imgfilepath.split('.')[-1])

        else:
            res = dict()
            res['status'] = '500'
            res['msg'] = 'The file format is only jpg or png'

    def run(self):
        self.provide_automatic_option = False
        self.app.add_url_rule('/detect', None, self.detect, methods = [ 'POST' ] )
        self.app.add_url_rule('/get_predict_image', None, self.get_predict_image, methods = [ 'POST' ] )

        print("server run")
        self.app.run(host=self.host, port=self.port)

def importargs():
    parser = argparse.ArgumentParser('This is a server of darknet')

    parser.add_argument("--cfgfilepath", "-cf", help = "config filepath  of darknet", type=str)
    parser.add_argument("--datafilepath", "-df", help = "datafilepath of darknet", type=str)
    parser.add_argument("--weightfilepath", "-wf", help = "weight filepath of darknet")
    parser.add_argument("--host", "-H", help = "host name running server",type=str, required=False, default='localhost')
    parser.add_argument("--port", "-P", help = "port of runnning server", type=int, required=False, default=8080)
    parser.add_argument("--uploaddir", "-ud", help = "upload folder of images")

    args = parser.parse_args()

    file_exist_flag = True

    if args.cfgfilepath:
        assert os.path.exists(args.cfgfilepath), "cfgfilepath of %s does not exist" % args.cfgfilepath
    else:
        file_exist_flag = False
        print("cfgfilepath is needed")

    if args.datafilepath:
        assert os.path.exists(args.datafilepath), "datafilepath of %s does not exist" % args.datafilepath
    else:
        file_exist_flag = False
        print("datafilepath is needed")

    if args.weightfilepath:
        assert os.path.exists(args.weightfilepath), "weightfilepath of %s does not exist" % args.weightfilepath
    else:
        file_exist_flag = False
        print("weightfilepath is needed")

    if args.uploaddir:
        assert os.path.exists(args.uploaddir) & os.path.isdir(args.uploaddir), "uploaddir of %s does not exist or is not directory" % args.uploaddir
    else:
        file_exist_flag = False
        print("uploaddir is needed")

    if file_exist_flag is False:
        parser.print_usage()
        sys.exit(1)

    return args.cfgfilepath, args.datafilepath, args.weightfilepath, args.host, args.port, args.uploaddir


def main():
    cfgfilepath, datafilepath, weightfilepath, host, port, uploaddir = importargs()
    yolo = Yolo(cfgfilepath, weightfilepath, datafilepath)
    server = MyServer('yolo_server', host, port, uploaddir, [ 'jpg', 'png' ], yolo )
    server.setup_converter()
    server.run()

if __name__ == "__main__":
    main()
