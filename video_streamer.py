from imutils.video.pivideostream import PiVideoStream
import time
import datetime
import numpy as np
import cv2


class VideoStreamer(object):
    def __init__(self, flip = False, resolution = (640, 480)):
        self.video_stream = PiVideoStream(resolution=resolution, framerate=8).start()
        time.sleep(2.0)

    def __del__(self):
        self.video_stream.stop()

    def get_frame(self):
        frame = self.video_stream.read()
        #ret, jpg = cv2.imencode('.jpg', frame)  # 圧縮
        return frame
        
