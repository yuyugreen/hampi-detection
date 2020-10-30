from flask import Flask, render_template, Response
from flask_sslify import SSLify
import cv2
from datetime import datetime
import threading
import requests
import time
import ssl
import numpy as np
import os

from video_streamer import VideoStreamer

camera = VideoStreamer()

app = Flask(__name__)
context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
context.load_cert_chain('/home/pi/cert.crt', '/home/pi/server_secret.key')

# 物体検出による位置の情報を、入力画像の座標に
# 物体検出モデルは(300, 300)
def scale_bounding_box_coordinate(image, target_bbox):
    # 画像の縦サイズ(shape[0])と横サイズ(shape[1])を取得
    image_height, image_width = image.shape[:2]

    # 予測値に元の画像サイズを掛けて、四角で囲むための4点の座標情報を得る
    coordinates = target_bbox[3:7] * (image_width, image_height, image_width, image_height)
    coordinates = coordinates.astype(np.int)[:4]  # 画像に四角や文字列を書き込むには、座標情報はintで渡す必要がある。

    return coordinates

# クラス名と共にBoundingBoxを画像に描画
def draw_bounding_box_on_frame(image, target_bbox, class_name):
    coordinates = scale_bounding_box_coordinate(image, target_bbox)

    # floatからintに変換して、変数に取り出す。
    # top_leftみたいな表記に変える？
    (start_X, start_Y, end_X, end_Y) = coordinates

    # BoundingBoxを描画する
    # (画像、開始座標、終了座標、色、線の太さ)を指定
    # OpenCVの関数を使うため、色の指定はBGRの順で行うことに留意
    cv2.rectangle(image, (start_X, start_Y), (end_X, end_Y), (255, 51, 51), thickness=2)

    # (画像、文字列、開始座標、フォント、文字サイズ、色を指定
    cv2.putText(image, class_name, (start_X, start_Y), cv2.FONT_HERSHEY_SIMPLEX, (.003*image.shape[1]), (255, 51, 51))

    return image

# LINE Notify APIを用いて、ローカルに保存されている画像のパスを参照し投稿する
def post_image_to_line_notify(line_token, message, image_path, line_api_url):
    line_header = {'Authorization': 'Bearer ' + line_token}
    line_post_data = {'message': message}
    line_image_file = {'imageFile': open(image_path, 'rb')}
    res = requests.post(line_api_url, data=line_post_data, 
                        headers=line_header, files=line_image_file)
    print(res.text)

# detectionには[?,id番号、予測確率、Xの開始点、Yの開始点、Xの終了点、Yの終了点]が入っている。
def detect_target_object_box(image, model, model_input_size=(300, 300)):
    # OpenCVはBGRで3チャンネルのカラー画像を扱っているため、ここでswapRBをTrueにして入力画像をRGBの順に変換
    model.setInput(cv2.dnn.blobFromImage(image, size=model_input_size, swapRB=True))

    # モデルのネットワークに対しフォワード処理を実施、推論結果を受け取る
    model_outputs = model.forward()

    # model_outputsは[1:1:100:7]のリストになっているため、後半の2つを取り出す
    detected_boxes = model_outputs[0, 0, :, :]  # 3次元目にクラスID、４次元目にモデルが予測した確率

    return detected_boxes


@app.route('/')
def index():
    return render_template('index.html')

# ストリーミングしつつ、物体検出したらLINEへ通知
def generate(camera):    
    PET_CLASS_ID = 1
    WAIT_SECOND = 600

    # 物体検出モデルのクラスidと対象物体名の辞書
    class_id_name_dict = {
        #0:'background',
        1:'hamster',
        2:'wheel',
        3:'toilet'
    }

    # LINE Notifyとの連携に関する情報
    LINE_API_URL = 'https://notify-api.line.me/api/notify'
    LINE_API_TOKEN = os.environ['LINE_API_TOKEN']
    MESSAGE = 'ハムスターが動きました🐹'

    CONFIDENCE = .5

    last_post_time = datetime(2000, 1, 1)  # このタイミングで適当な日付で初期化

    # Tensorflow Object Detection APIで訓練した物体検出モデルの読み込み
    # models/hogeフォルダ以下に、該当モデルのpb及びpbtxtファイルを格納する
    model_name = '20200926_ssd_mobilenet_v2_momentum_no_transfer_hmn'
    model = cv2.dnn.readNetFromTensorflow('models/{}/frozen_inference_graph.pb'.format(model_name),
                                        'models/{}/frozen_inference_graph.pbtxt'.format(model_name))

    # 毎フレームに対して行う処理
    while True:
        frame = camera.get_frame()  # フレームをカメラモジュールから取得

        # フレームから物体検出した推論結果を受け取る。
        # detectionには[?,id番号、予測確率、Xの開始点、Yの開始点、Xの終了点、Yの終了点]が入っている。
        detected_boxes = detect_target_object_box(frame, model)

        # 検出したBoundingBoxのうち、予測確率が最も高い枠だけ残す。
        target_bboxes = {}
        for box in detected_boxes:
            for class_id in class_id_name_dict.keys():   
                if (box[1] == class_id) & (box[2] >= CONFIDENCE):
                    target_bboxes[class_id] = box

        # フレーム内に対象物体(class_id==1)が検出されたとき、物体の枠をストリーミングに描画し、LINEにそのフレーム画像を投稿
        if PET_CLASS_ID in target_bboxes:
            # クラス名と共にBoundingBoxを描画
            frame = draw_bounding_box_on_frame(frame, target_bboxes[PET_CLASS_ID], class_id_name_dict[PET_CLASS_ID])
            
            # 最後の投稿からWAIT_SECOND以上経っている場合、LINEへ物体検出時のフレーム画像を投稿
            now = datetime.now()
            if ((now - last_post_time).total_seconds() > WAIT_SECOND): 
                # まずLINEへ投稿するフレームをローカルに保存
                image_path = 'img/{}.jpg'.format(now.strftime('%Y%m%d%H%M%S'))
                cv2.imwrite(image_path, frame)

                # LINEへ投稿
                post_image_to_line_notify(LINE_API_TOKEN, MESSAGE, image_path, LINE_API_URL)
                last_post_time = now

            # GCPのIoT Coreへログを送信
            frame_height, frame_width = frame.shape[:2]
            for class_id in target_bboxes.keys():
                #class_name = class_id_name_dict[class_id]
                start_x = target_bboxes[class_id][3]
                start_y = target_bboxes[class_id][4]
                end_x = target_bboxes[class_id][5]
                end_y = target_bboxes[class_id][6]
                cmd = 'cd /home/pi/iotcore/; java -jar raspi-comfort-sensor-iotcore-1.0.jar ' 
                opt = '--start_x {} --start_y {} --end_x {} --end_y {} --frame_height {} --frame_width {}'  \
                .format(start_x, start_y, end_x, end_y, frame_height, frame_width)

                os.system(cmd + opt)

        # ストリーミングに向けた型変換
        frame_encode = cv2.imencode('.jpg',frame)[1]
        string_frame_data = frame_encode.tostring()

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + string_frame_data + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate(camera),  # generate関数からframeをストリーム
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, ssl_context=context, threaded=True, debug=False)
