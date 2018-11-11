#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import configparser
import datetime
import io
import sys
import os
import re


import cv2
from flask import Flask, request, redirect, jsonify
from flask import send_file
import numpy as np
from PIL import Image
from pykakasi import kakasi
from werkzeug import secure_filename


from darknet import Darknet
from yolo_result import YoloResult

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
            # for debug
            output_filename = "%s_%s" % (datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), self.convert_filename(file.filename))
            print("output filename is %s" % output_filename)
            outputfilepath = os.path.join(self.config['UPLOAD_FOLDER'], output_filename)
            file.save(outputfilepath)

            try:
                # ToDo: use io.BytesIO
                img = np.array(Image.open(outputfilepath))
            except Exception as err:
                print(err)

            if request.form.get("thresh"):
                thresh = float(request.form.get("thresh"))
                print("the request parameter of thresh hold is %f" % thresh)
                yolo_results = self.yolo.detect(img, thresh)
            else:
                print("the threshold is not included of parameter")
                yolo_results = self.yolo.detect(img)
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
                    cv2_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    pred_img = self.yolo.draw_detections(cv2_img, yolo_results)
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
            cv2_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            pred_img = self.yolo.draw_detections(cv2_img, yolo_results)

            tmpfilename = outputfilepath.split(os.path.sep)[-1]
            print(tmpfilename)
            pred_outputfilename = "%s_pred.jpg" % tmpfilename.split('.')[0]
            pred_img_outputfilepath = os.path.join(self.config['UPLOAD_FOLDER'], pred_outputfilename)

            print("pred img outputfilepath: %s" % pred_img_outputfilepath)
            cv2.imwrite(pred_img_outputfilepath, pred_img)

            with open(pred_img_outputfilepath, 'rb') as img:
                return send_file(io.BytesIO(img.read()),
                        attachment_filename=pred_outputfilename,
                        mimetype='image/%s' % pred_outputfilename.split('.')[-1])

        else:
            res = dict()
            res['status'] = '500'
            res['msg'] = 'The file format is only jpg or png'


def get_params(configfilepath):
    """
    getting parameter from config
    """

    config = configparser.ConfigParser()
    config.read(configfilepath)

    try:

        # for YOLO
        darknetlibfilepath = config.get('YOLO', 'darknetlibfilepath')
        datafilepath = config.get('YOLO', 'datafilepath')
        cfgfilepath = config.get('YOLO', 'cfgfilepath')
        weightfilepath = config.get('YOLO', 'weightfilepath')

        # for Server
        host = config.get('Server', 'host')
        port = config.getint('Server', 'port')
        uploaddir = config.get('Server', 'uploaddir')
        pub_img_flg = config.getboolean('Server', 'publish_image_flg')
        return darknetlibfilepath, datafilepath, cfgfilepath, \
            weightfilepath, host, port, uploaddir, pub_img_flg

    except configparser.Error as config_parse_err:
        raise config_parse_err

def check_path(targetpath):
    """
    checking path
    """
    if not os.path.exists(targetpath):
        print('%s does not exist' % targetpath)
        return False
    else:
        return True


def validate_host_name(hostname):
    """
    validate host name
    refference: https://stackoverflow.com/questions/2532053/validate-a-hostname-string
    """

    if len(hostname) > 255:
        return False
    if hostname[-1] == ".":
        hostname = hostname[:-1] # strip exactly one dot from the right, if present
    allowed = re.compile("(?!-)[A-Z\d-]{1,63}(?<!-)$", re.IGNORECASE)
    return all(allowed.match(x) for x in hostname.split("."))


def check_params(darknetlibfilepath, datafilepath, cfgfilepath, 
                 weightfilepath, host, port, uploaddir, pub_img_flg):
    """
    checking parameters
    """

    validation_flg = True
    for targetpath in [darknetlibfilepath, datafilepath,
                       cfgfilepath, weightfilepath]:
        validation_flg = check_path(targetpath)

    if not validate_host_name(host):
        print('%s is invalid as host name' % host)
    
    if port < 0 or port > 65535:
        print('port should be between 0 and 65535 but actual is %d' % host)
        validation_flg = False
    elif port > 0 and port < 1024:
        print('[Waring] you use well-known ports, %d' % port)

    return validation_flg

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
    darknet_server_conffilepath = "./conf/darknet_server.ini"
    darknetlibfilepath, datafilepath, cfgfilepath, \
        weightfilepath, host, port, uploaddir, \
            pub_img_flg = get_params(darknet_server_conffilepath)

    if check_params(darknetlibfilepath, datafilepath, cfgfilepath,
                    weightfilepath, host, port, uploaddir, pub_img_flg):

        darknet = Darknet(libfilepath=darknetlibfilepath,
                          cfgfilepath=cfgfilepath.encode(),
                          weightsfilepath=weightfilepath.encode(),
                          datafilepath=datafilepath.encode())

        darknet.load_conf()


        server = DarknetServer(host, 'darknet_server',
                               uploaddir, ['jpg', 'png'],
                               pub_img_flg, darknet)
        server.setup_converter()
        print("server run")
        server.run(host=host, port=port)
    else:
        print('validation error')
        sys.exit(1)


if __name__ == "__main__":
    main()
