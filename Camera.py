import cv2
import numpy as np


class Camera(object):
    """普通摄像头
    """
    def __init__(self, id=0, width=640, height=480):
        self.cap = cv2.VideoCapture(id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # 设置摄像头采集图像分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def get_data(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                return frame

    def __del__(self):
        self.cap.release()

if __name__ == "__main__":
    cam = Camera(0)

    while True:
        color = cam.get_data( )

        cv2.namedWindow('color', 0)
        cv2.imshow('color', color)
        c = cv2.waitKey(1)
        if c == 27: # 按下esc退出
            exit()