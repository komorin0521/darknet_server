# Overview
This is the python script, which is server of detection using darknet yolo.

Darknet is as following:
 - https://pjreddie.com/darknet/yolo/

This server has two API.

1. Getting the detection result as json format
    URI: '/detect'

2. Getting the image which is embedded the rectangle and object name
    URI: '/get_predict_image'

# SoftWare
- Python: 2.7.x

    I test only 2.7.13

- darknet revision: 1b001a7f58aacc7f8b751332d3a9d6d6d0200a2d

# Setup
1. Install darknet
   Install darknet, reading the website of darknet

2. Clone of this repository

    `$ git clone https://github.com/komorin0521/darknet_server`

3. Move all files and folders into darknet folder

    `$ cp -r darknet_server/* darknet/`

4. Install python module using pip

    `$ (sudo) pip install -r requirements.txt`

5. Running server

    `$ python server.py -cf ./cfg/yolo.cfg -df ./cfg/coco.data -wf ./yolo.weights -ud ./upload`

6. Check the server response from other terminal
    - Getting detection result

        `$ curl -XPOST -F file=@./data/person.jpg http://localhost:8080/detect`

        If the server work well, you will get message like following

        ```
        {
            'status' : 200,
            'result' : [
                            { 'obj_name' : 'dog',
                              'score' : 0.86223...,
                     'bounding_box' : {
                        'height' : 86.09...,
                        'width' : 137.617...,
                        'x_min': '61.686...',
                        'y_min': '264.982...'

                     }
                   },
                   {
                     ...
                   }
                 ]
        }
        ```

    - Getting image

        `$ curl -XPOST -F file=@/home/omori/darknet/data/person.jpg http://localhost:8080/get_predict_image > predictions.jpg`

# More information
If you want more information about scripts, please show the help as following.

`$ python server.py -h`
