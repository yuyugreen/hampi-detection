from flask import Flask, render_template, Response
from flask_sslify import SSLify
import cv2
from datetime import datetime
import threading
import requests
import time
import ssl
import os

from video_streamer import VideoStreamer

app = Flask(__name__)
context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
context.load_cert_chain('/home/pi/cert.crt', '/home/pi/server_secret.key')

camera = VideoStreamer()

# LINE Notify トークン
LINE_API_URL = 'https://notify-api.line.me/api/notify'
LINE_API_TOKEN = os.environ['LINE_API_TOKEN']

@app.route('/')
def index():
    return render_template('index.html')

# ストリーミング
def generate():
    while True:
        frame_stream = camera.get_frame()  # カメラモジュールからフレームを取得

        # グレースケール化
        frame_stream_gray = cv2.cvtColor(frame_stream, cv2.COLOR_BGR2GRAY)

        # ストリーミングのために型変換
        frame_encode = cv2.imencode('.jpg',frame_stream_gray)[1]
        string_frame_data = frame_encode.tostring()

        time.sleep(0.5)

        # Escキーで終了
        key = cv2.waitKey(33)
        if key == 27:
            break

        # frameをストリーム
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + string_frame_data + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate(),  # generate関数からframeをストリーム
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# 輪郭を絞り込む関数（サイズで絞り込み）
def extract_rectangles_from_contours(contours, min_figure_size):
    '''
    contours:領域の四点のx,y座標
    size:どのくらいのサイズ以上だったら抽出するのか、という閾値
    返り値:(左上の x 座標, 左上の y 座標, 幅, 高さ) であるタプル
    '''
    list_extracted_rectangles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_figure_size:
            rectangle = cv2.boundingRect(cnt)
            list_extracted_rectangles.append(rectangle)
    
    return list_extracted_rectangles

@app.route('/line_notify')
def line_notify():
    # frameの移動平均を計算、ある程度の大きさの動体を検知したらLINEへ通知＆動画保存、そのフレームを投稿する。
    frame_mov_avg = None
    while True:
        frame = camera.get_frame()  # カメラモジュールからフレームを取得
        now = datetime.now()  # 現在時刻のdatetimeを取得

        # 明暗変化による動体誤判定を防ぐために、グレースケール化
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # frameの移動平均を初期化
        if frame_mov_avg is None:
            frame_mov_avg = frame_gray.copy().astype("float")
            last_post_time = datetime(2000, 1, 1)  # このタイミングで適当な日付で初期化
            continue

        # 現在のフレームと前フレーム間の移動平均を計算
        cv2.accumulateWeighted(frame_gray, frame_mov_avg, 0.01)
        frameDelta = cv2.absdiff(frame_gray, cv2.convertScaleAbs(frame_mov_avg))

        # 画素値の閾値を設定し、フレームを白黒に2値化
        thresh = cv2.threshold(frameDelta, 15, 255, cv2.THRESH_BINARY)[1]
        
        # 画像内の差分部分（動きがあった部分）の輪郭を見つける
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        # 輪郭で囲まれた図形を「ある程度以上の大きさのもの」に絞り込み、listに格納
        min_figure_size = 500
        list_extracted_rectangles = extract_rectangles_from_contours(contours, min_figure_size)
        
        # frame内に動体が検出された場合の処理
        if list_extracted_rectangles != []:  # frame内に一定の大きさの図形がある
            # 動体検出時点のframeをjpgに保存
            post_jpg = now.strftime('%Y%m%d%H%M%S') + '.jpg'
            cv2.imwrite('img/'+post_jpg, frame)
            
            # 動体検出後の処理
            notify_interval = 10  # デフォルトの通知間隔は10分に設定する
            notify_interval_sec = notify_interval * 60  # 10分間（秒に換算）
            # 動体検出時（前回の録画開始10分後に動体を検出した時=list_extracted_rectanglesに何かが入った時）
            if ((now - last_post_time).total_seconds() > notify_interval_sec):  # LINEへのlast_post_timeをフラグとする
                # LINEへ通知を送る　# LINEへのlast_post_timeから10分以上経っている場合、送信
                line_header = {'Authorization': 'Bearer ' + LINE_API_TOKEN}
                line_post_data = {'message': 'ハムスターが動きました🐹'}
                line_image_file = {'imageFile': open('img/'+post_jpg, 'rb')}  # 動体の枠付きのframe
                res = requests.post(LINE_API_URL, data=line_post_data, 
                                    headers=line_header, files=line_image_file)
                last_post_time = now
                print(res.text)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, ssl_context=context, threaded=True, debug=False)
