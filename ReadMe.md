# Overview
This is the python script, which is server of detection using darknet yolo.

Darknet is as following:
 - https://pjreddie.com/darknet/yolo/

This server has two API.

1. Getting the detection result as json format including image(base64 decoded)
    URI: '/detect'

    parameter example
    ```
    {
      "image": base64.b64decode(imgdata).decode("utf-8"),
      "thesh": 0.5,
      "get_img_flg": True
    ```

    The parameter of 'thresh' is threshold of yolo.
    This is not required and the default value is 0.5.
    If thresh is less than default, more object can be detected but misrecgnition increses.

    The paramete of `get_img_flg` is boolean.
    If true, return image data embeded bbox of yolo resuls.
    This parameter is optional and the defalut value is true.

# SoftWare requirements
- Python > 3.6

    I test only 3.7.1

- darknet revision: 9a4b19c4158b064a164e34a83ec8a16401580850
  darknet revision: 61c9d02ec461e30d55762ec7669d6a1d3c356fb2

# Setup
## Install Directly Your Machine
1. Install darknet
   Install darknet, reading the website of darknet

2. Clone of this repository

    `$ git clone https://github.com/komorin0521/darknet_server`

3. Move all files and folders into darknet folder

    `$ cp -r darknet_server/* ${darknetPATH}/python`

4. Install python module using pip

    `$ (sudo) pip install -r requirements.txt`

5. Running server

    `$ python3 darknet_server.py`


If you want to your original model,
please modify 'darknet_server/conf/darknet_server.ini'


## Using docker

### requirements
1. docker-compose version 1.19.0-rc3, build cdd0282
2. Docker version 18.06.1-ce, build e68fc7a. 
    installed `nvidia-docker`

I use the cuda-9.2 and nvidia driver is 396.44
If you use another version, please modify the source image of dokcer,
"nvidia/cuda:9.2-devel-ubuntu16.04"


### How to install
1. install docker, nvidia-docker, docker-compose
2. please check the run of `nvidia-smi`
3. start docker

   ```
   $ docker-compose up
   ```

# Check the server response from other terminal

```
$ python simple_client.py |sed "s/'/\"/g" |jq
```

If the server work well, you will get message like following

```
{
  "status": "success",
  "resultlist": [
    {
      "class_index": 0,
      "obj_name": "person",
      "score": 0.9997863173484802,
      "bounding_box": {
        "x_min": 191.12104034423828,
        "y_min": 100.11891174316406,
        "width": 79.28132629394531,
        "height": 277.3841247558594
      }
    },
    {
      "class_index": 17,
      "obj_name": "horse",
      "score": 0.9993027448654175,
      "bounding_box": {
        "x_min": 402.7898635864258,
        "y_min": 135.62349700927734,
        "width": 198.86619567871094,
        "height": 219.7958526611328
      }
    },
    {
      "class_index": 16,
      "obj_name": "dog",
      "score": 0.9974156618118286,
      "bounding_box": {
        "x_min": 60.97343444824219,
        "y_min": 264.89939880371094,
        "width": 138.28082275390625,
        "height": 82.55728149414062
      }
    }
  ]
}
```

# More information
If you want more information about scripts, please show the help as following.

`$ python server.py -h`

# Contacts
If you have any questions, please send me an email or pull request.

email: yu.omori0521@gmail.com
