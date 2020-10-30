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

# for yolov5 import
import sys
sys.path.append("/home/pi/yolov5")
import torch

from utils.datasets import *
from utils.utils import *

# model settings
device = 'cpu'
weights = 'models/best.pt' # self trained model
model = torch.load(weights, map_location=device)['model'].float()

# Get names and colors
names = model.names if hasattr(model, 'names') else model.modules.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

camera = VideoStreamer()

# Flask settings
app = Flask(__name__)
context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
context.load_cert_chain('/home/pi/cert.crt', '/home/pi/server_secret.key')

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

# LINE Notify APIを用いて、ローカルに保存されている画像のパスを参照し投稿する
def post_image_to_line_notify(line_token, message, image_path, line_api_url):
    line_header = {'Authorization': 'Bearer ' + line_token}
    line_post_data = {'message': message}
    line_image_file = {'imageFile': open(image_path, 'rb')}
    res = requests.post(line_api_url, data=line_post_data, 
                        headers=line_header, files=line_image_file)
    print(res.text)

def detect_bboxes(frame):
    img = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
    img = letterbox(img, new_shape=320)[0]
    img = np.transpose(img, (2,0,1))

    img = torch.from_numpy(img).to(device)

    img = img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=False)[0]
    # Apply NMS
    # detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    pred = non_max_suppression(pred, 0.4, 0.5, classes=None, agnostic=True)
    dets = pred[0]
    if dets is not None: 
        # Rescale boxes from image_size to frame_size
        dets[:, :4] = scale_coords(img.shape[2:], dets[:, :4], frame.shape).round()
        dets = dets.detach().numpy()
    return dets

@app.route('/')
def index():
    return render_template('index.html')

# ストリーミングしつつ、物体検出したらLINEへ通知
def generate(camera):    
    WAIT_SECOND = 600

    # LINE Notifyとの連携に関する情報
    LINE_API_URL = 'https://notify-api.line.me/api/notify'
    LINE_API_TOKEN = os.environ['LINE_API_TOKEN']
    MESSAGE = 'ハムスターが動きました🐹'

    CONFIDENCE = .7

    last_post_time = datetime(2000, 1, 1)  # このタイミングで適当な日付で初期化

    # 毎フレームに対して行う処理
    while True:
        frame = camera.get_frame()  # フレームをカメラモジュールから取得

        # フレームから物体検出した推論結果を受け取る。
        bboxes = detect_bboxes(frame)
        print(bboxes)       

        if bboxes is not None:
            highest_score = CONFIDENCE
            bbox = None
            # select highest scored box
            if np.all(bboxes[:, 5]!=1):  # no hand
                for b in bboxes:
                    if (b[5] == 0) & (b[4] > highest_score):
                        bbox = b
                        highest_score = b[4]

            if bbox is not None:
                # 物体の枠をストリーミングに描画
                label = '%s %.2f' % (names[int(bbox[5])], bbox[4])  # class_number, confidence
                plot_one_box(bbox[:5], frame, label=label, color=colors[int(bbox[4])])
                
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
                # TODO:Add conf, class_number  
                frame_height, frame_width = frame.shape[:2]
                start_x = bbox[0]
                start_y = bbox[1]
                end_x = bbox[2]
                end_y = bbox[3]
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
