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

# SoftWare requirements
- Python: 3.x

    I test only 3.5.2

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
