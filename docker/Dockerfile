FROM nvidia/cuda:9.2-devel-ubuntu16.04
RUN set -x && \
    apt update && \
    apt install -y --no-install-recommends \
        libopencv-dev python3-dev python3-pip python3-setuptools git wget
WORKDIR /opt
RUN git clone https://github.com/pjreddie/darknet
WORKDIR /opt/darknet
RUN git checkout 61c9d02ec461e30d55762ec7669d6a1d3c356fb2
RUN sed -i -e "s/GPU\=0/GPU\=1/" -e "s/OPENCV\=0/OPENCV\=1/g" Makefile
RUN make
WORKDIR /opt/
RUN git clone https://github.com/komorin0521/darknet_server
WORKDIR /opt/darknet_server
RUN git checkout devel
RUN pip3 install pip --upgrade
RUN pip3 install --no-cache-dir -r ./requirements.txt
RUN ln -sf /opt/darknet/data ./
WORKDIR /tmp/
RUN wget https://pjreddie.com/media/files/yolov3.weights
WORKDIR /opt/darknet_server
CMD ["python3", "./darknet_server.py"]
