#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
darknet yolo server
"""

import base64
import configparser
import io
import logging
import re
import sys
import os

import cv2
import numpy as np
from PIL import Image
import responder


from darknet import Darknet


class DarknetServer:
    """
    Darknet Yolo server
    """

    def __init__(self, host, port, upload_dir, yolo, logger):
        """
        init server class
        """
        self.api = responder.API()
        self.host = host
        self.port = port
        self.upload_dir = upload_dir
        self.yolo = yolo
        self.define_uri()
        self.logger = logger

    def define_uri(self):
        """
        definition of uri
        """
        self.api.add_route('/detect', self.detect)

    def get_yolo_results(self, img_data, thresh=0.5):
        """
        get result
        """
        self.logger.info("call get_result")

        try:
            img_np_arr = np.array(Image.open(
                io.BytesIO(img_data)).convert("RGB"))
            self.logger.info("call self.yolo.detect")
            yolo_results = self.yolo.detect(img_np_arr, thresh)
            for yolo_result in yolo_results:
                self.logger.info(yolo_result.get_detect_result())
        except Exception as err:
            raise err

        return img_np_arr, yolo_results

    async def detect(self, req, resp):
        """
        Detection using yolo. '/detect'
        """
        self.logger.info("call api of detect")

        req_data = await req.media()

        if "image" in req_data:
            self.logger.info("req has key of image")
            img_data = base64.b64decode(req_data["image"])
            if "thresh" in req_data:
                thresh = req_data["thresh"]
            else:
                thresh = 0.5
            try:
                img_np_arr, yolo_results = self.get_yolo_results(img_data,
                                                                 thresh)
                res = {"status": "success",
                       "resultlist": [yolo_result.get_detect_result() for yolo_result in yolo_results]}

                if "get_img_flg" in req_data:
                    get_img_flg = req_data["get_img_flg"]
                else:
                    get_img_flg = True

                if get_img_flg:
                    cv2_img = cv2.cvtColor(img_np_arr, cv2.COLOR_RGB2BGR)
                    cv2_img = self.yolo.draw_detections(cv2_img,
                                                        yolo_results)
                    _, buf = cv2.imencode(".png", cv2_img)
                    res["pred_img"] = base64.b64encode(buf).decode("utf-8")
                resp.media = res

            except Exception as err:
                self.logger.error(err)
                resp.media = {"status": "fatal",
                              "msg": "An error occured in server"}

        else:
            resp.media = {"status": "fatal",
                          "msg": "request should have the key of image"}

    def run_server(self):
        """
        run server
        """
        self.api.run(address=self.host,
                     port=self.port,
                     logger=self.logger)


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
        logfilepath = config.get('Server', 'logfilepath')
        uploaddir = config.get('Server', 'uploaddir')
        return darknetlibfilepath, datafilepath, cfgfilepath, \
            weightfilepath, host, port, logfilepath, uploaddir
    except configparser.Error as config_parse_err:
        raise config_parse_err


def check_path(targetpath):
    """
    checking path
    """
    check_flg = None
    if not os.path.exists(targetpath):
        print('%s does not exist' % targetpath)
        check_flg = False
    else:
        check_flg = True
    return check_flg


def validate_host_name(hostname):
    """
    validate host name
    refference:
    https://stackoverflow.com/questions/2532053/validate-a-hostname-string
    """
    valid_flg = True
    if len(hostname) > 255:
        valid_flg = False
    if hostname[-1] == ".":
        # strip exactly one dot from the right, if present
        hostname = hostname[:-1]
        allowed = re.compile(r"(?!-)[A-Z\d-]{1,63}(?<!-)$", re.IGNORECASE)
        valid_flg = all(allowed.match(x) for x in hostname.split("."))
    return valid_flg

def check_params(darknetlibfilepath, datafilepath, cfgfilepath,
                 weightfilepath, host, port, logger):
    """
    checking parameters
    """

    validation_flg = True
    for targetpath in [darknetlibfilepath, datafilepath,
                       cfgfilepath, weightfilepath]:
        validation_flg = check_path(targetpath)

    if not validate_host_name(host):
        logger.error('%s is invalid as host name' % host)

    if port < 0 or port > 65535:
        logger.error('port should be between 0 and 65535 but actual is %d'
                     % host)
        validation_flg = False
    elif 0 < port < 1024:
        logger.warning('[Waring] you use well-known ports, %d' % port)

    return validation_flg


def main():
    """
    Main
    """

    darknet_server_conffilepath = "./conf/darknet_server.ini"
    darknetlibfilepath, datafilepath, cfgfilepath, \
        weightfilepath, host, port, logfilepath, \
        uploaddir = get_params(darknet_server_conffilepath)

    logging.basicConfig(filename=logfilepath,
                        format="[%(asctime)s]\t[%(levelname)s]\t%(message)s",
                        level=logging.INFO)
    logger = logging.getLogger("darknet_server")

    logger.info("libfilepath: %s", darknetlibfilepath)
    logger.info("datafilepath: %s", datafilepath)
    logger.info("cfgfilepath: %s", cfgfilepath)
    logger.info("weightfilepath: %s", weightfilepath)
    logger.info("host: %s", host)
    logger.info("port: %d", port)

    if check_params(darknetlibfilepath, datafilepath, cfgfilepath,
                    weightfilepath, host, port, logger):

        darknet = Darknet(libfilepath=darknetlibfilepath,
                          cfgfilepath=cfgfilepath.encode(),
                          weightsfilepath=weightfilepath.encode(),
                          datafilepath=datafilepath.encode())

        darknet.load_conf()

        server = DarknetServer(host, port, uploaddir, darknet, logger)
        logger.info("server run")
        server.run_server()
    else:
        logger.error('validation error')
        sys.exit(1)


if __name__ == "__main__":
    main()
