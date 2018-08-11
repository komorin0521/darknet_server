# Overview
darknet(yolo)をサーバ化したpythonのソースです。

darknetについては下記を見てください
- https://pjreddie.com/darknet/yolo/


本サーバでは2つのAPIを実装しています。

1. 検出結果取得API
    URI: '/detect'
    パラメータは"file"と"thresh"（閾値）です。
    閾値は任意で、デフォルトはdarknet本家と同じ0.25にしてます。
    低く設定するとその分、より多くの検出結果が返ってきますが、
    誤認識も多くなります。

2. 検出結果の画像取得API
    URI: '/get_predict_image'
    パラメータは検出結果取得APIと同様です

# SoftWare
- Python: 3.5.x

    3.5.2でのみ動作確認しています

- darknet revision

    9a4b19c4158b064a164e34a83ec8a16401580850

    上記のコミットでのみ動作確認しています。

# Setup
1. darknetを本家のサイトに従いインストールする

2. 本レポジトリをcloneする

    `$ git clone https://github.com/komorin0521/darknet_server`

3. 本レポジトリのファイルをdarknetと同様のフォルダに移動させる

    `$ cp -r darknet_server/* darknet/python`

4. pipを用いて必要なパッケージをインストールする

    `$ (sudo) pip install -r requirements.txt`

5. サーバを起動する

    `$ PYTHONPATH=${darknet_path}/python python python/darknet_server.py -cf ./cfg/yolov3.cfg -df ./cfg/coco.data -wf ./yolov3.weights -ud ./upload -pf false`

    You can show the detail of arguments using option of "-h".

6. 別ターミナルを開いて結果を確認する
    - 検出結果取得API

        `$ curl -XPOST -F file=@./data/person.jpg http://localhost:8080/detect`

        正しく動作していた場合、下記のような結果が返ってきます。
        "-pf"オプションがtrueかつ'/var/www/html/images'に対してpythonプロセスの実行ユーザに書き込み権限があれば
        画像パスも返します。

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

        異なる閾値で結果を取得する場合は下記のようにリクエストしてください

        `curl -F file=@./data/person.jpg -F thresh=0.850 http://localhost:8080/detect`

    - 検出画像取得API

        `$ curl -XPOST -F file=@/home/omori/darknet/data/person.jpg http://localhost:8080/get_predict_image > predictions.jpg`

        異なる閾値で結果を取得する場合は下記のようにリクエストしてください

        `curl -F file=@./data/person.jpg -F thresh=0.850 http://localhost:8080/get_predict_image > predictions.jpg`

# More information
引数の詳細等については下記のコマンドを実行してください

`$ python darknet_server.py -h`

# Contacts
ご質問等があればpullリクエストやメールください

email: yu.omori0521@gmail.com
