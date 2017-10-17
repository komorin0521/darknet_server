#!/usr/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import


import argparse
import sys
import os

from flask import Flask, request, redirect, jsonify
from werkzeug import secure_filename



sys.path.append(os.path.join(os.getcwd(),'python/'))
import darknet as dn
from yolo import Yolo
from yolo import YoloResult

class MyServer(object):
    def __init__(self, name, host, port, upload_dir, extensions, yolo):
        self.app = Flask(name)
        self.host = host
        self.port = port
        self.app.config['UPLOAD_FOLDER'] = upload_dir
        self.extensions = extensions
        self.yolo = yolo

    def check_allowfile(self, filename):
        if len(filename.split(".")) > 1:
            return filename.split(".")[-1] in self.extensions
        else:
            return False

    def detect(self):
        print("call detect")
        if request.method == 'POST':
            file = request.files['file']
            if file and self.check_allowfile(file.filename):
                print("saving file")
                output_filename = secure_filename(file.filename)
                outputfilepath = os.path.join(self.app.config['UPLOAD_FOLDER'], output_filename)
                file.save(outputfilepath)
                yolo_results = self.yolo.detect(outputfilepath)

                res = dict()
                res['status'] = '200'
                res['result'] = list()
                for yolo_result in yolo_results:
                    # print(yolo_result.get_detect_result())
                    res['result'].append(yolo_result.get_detect_result())

                return jsonify(res)
            else:
                res = dict()
                res['status'] = '500'
                res['msg'] = 'The file format is only jpg or png'

    def run(self):
        self.provide_automatic_option = False
        self.app.add_url_rule('/detect', None, self.detect, methods = [ 'POST' ] )
        print("server run")
        self.app.run(host=self.host, port=self.port)


def importargs():
    parser = argparse.ArgumentParser('This is a server of darknet')

    parser.add_argument("--cfgfilepath", "-cf", help = "config filepath  of darknet", type=str)
    parser.add_argument("--datafilepath", "-df", help = "datafilepath of darknet", type=str)
    parser.add_argument("--weightfilepath", "-wf", help = "weight filepath of darknet")


    parser.add_argument("--host", "-H", help = "host name running server",type=str, required=False, default='localhost')

    parser.add_argument("--port", "-P", help = "port of runnning server", type=str, required=False, default='8080')

    parser.add_argument("--uploaddir", "-ud", help = "upload folder of images")

    args = parser.parse_args()

    assert os.path.exists(args.cfgfilepath), "cfgfilepath of %s does not exist" % args.cfgfilepath

    assert os.path.exists(args.datafilepath), "datafilepath of %s does not exist" % args.datafilepath

    assert os.path.exists(args.weightfilepath), "weightfilepath of %s does not exist" % args.weightfilepath

    assert os.path.exists(args.uploaddir) & os.path.isdir(args.uploaddir), "uploaddir of %s does not exist or is not directory" % args.uploaddir

    return args.cfgfilepath, args.datafilepath, args.weightfilepath, args.host, args.port, args.uploaddir


def main():
    cfgfilepath, datafilepath, weightfilepath, host, port, uploaddir = importargs()

    yolo = Yolo(cfgfilepath, weightfilepath, datafilepath)
    server = MyServer('yolo_server', 'localhost', '8080', './upload', [ 'jpg', 'png' ], yolo )
    server.run()


if __name__ == "__main__":
    main()
