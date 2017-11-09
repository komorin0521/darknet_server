# Overview
This is the python script, which is server of detection using darknet yolo.

Darknet is as following:
 - https://pjreddie.com/darknet/yolo/

This server has two API.

1. Getting the detection result as json format
    URI: '/detect'
    The parameter is 'thresh' which is threshold of yolo.
    This is not required and the default value is 0.25.
    If thresh is less than default, more object can be detected but misrecgnition increses.


2. Getting the image which is embedded the rectangle and object name
    URI: '/get_predict_image'
    The parameter is same as '/detect'

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

    `$ python server.py -cf ./cfg/yolo.cfg -df ./cfg/coco.data -wf ./yolo.weights -ud ./upload -pf false`

    You can show the detail of arguments using option of "-h".

6. Check the server response from other terminal
    - Getting detection result

        `$ curl -XPOST -F file=@./data/person.jpg http://localhost:8080/detect`

        If the server work well, you will get message like following
        When you start up "-pf true" and the process can access /var/www/html/images, 
        you can get the image src.

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
                 ],
            'img_src' : 'http://localhost/images/yyyymmdd_pred.jpg'
        }
        ```

        If you want to change the threshold, please request like following

        `curl -F file=@./data/person.jpg -F thresh=0.850 http://localhost:8080/detect`

    - Getting image

        `$ curl -XPOST -F file=@/home/omori/darknet/data/person.jpg http://localhost:8080/get_predict_image > predictions.jpg`

        If you want to change the threshold, please request like following

        `curl -F file=@./data/person.jpg -F thresh=0.850 http://localhost:8080/get_predict_image > predictions.jpg`

# More information
If you want more information about scripts, please show the help as following.

`$ python server.py -h`

# Contacts
If you have any questions, please send me an email or pull request.

email: yu.omori0521@gmail.com
