#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
simple client script
"""

import base64
import io

from PIL import Image
import requests


def main():
    """
    main
    """

    with open("./data/person.jpg", "rb") as inputfile:
        data = inputfile.read()

    post_data = {"image": base64.b64encode(data).decode("utf-8"),
                 "get_img_flg": True}
    res = requests.post("http://localhost:8080/detect", json=post_data).json()

    if "pred_img" in res:
        pred_data = base64.b64decode(res["pred_img"])
        img = Image.open(io.BytesIO(pred_data))
        img.save("test.png")
        del res["pred_img"]
        print(res)

if __name__ == "__main__":
    main()
