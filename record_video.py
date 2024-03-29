import os
import cv2
import numpy as np
from pathlib import Path
from cameras import KinectCamera as Camera
import time

BASE_PATH = Path(__file__).resolve().parent
IMG_PATH = BASE_PATH / 'images'

# init first img id
def increment_path(path, name='color_', suffix='png'):
    if not path.exists():
        path.mkdir()
    for n in range(0, 9999):
        if not os.path.exists(f'{path}\{name}{n:03d}.{suffix}'):  #
            break
    return n

IMG_ID = increment_path(IMG_PATH, name = f'color_')  # first id
PAUSE_FLAG = True
cnt = 10  # 拍摄图片数量，当为 0 时退出
interval = 5  # 间隔时间

# save data
def save_data(color_img, depth_img, id):
    cv2.imwrite(f'{IMG_PATH}\color_{id:03d}.png', color_img)
    cv2.imwrite(f'{IMG_PATH}\depth_{id:03d}.png', depth_img)
    print(f'Saved at {IMG_PATH}\color_{id:03d}')


#  click mouse mid button to pause
def onMouse(event, x, y, flags, param):
    global PAUSE_FLAG
    if event == cv2.EVENT_MBUTTONDOWN:
        PAUSE_FLAG ^= True

if __name__ == '__main__':
    last_time = time.time()
    cam = Camera()
    cv2.namedWindow('Kinect', 0)
    cv2.setMouseCallback('Kinect', onMouse)
    while True:
        color_img, depth_img, depth_colormap = cam.get_data()

        # Show images
        images = np.vstack([color_img[:,:,:3], depth_colormap])
        cv2.imshow('Kinect', images)
        c = cv2.waitKey(1)

        # record img
        if time.time() - interval > last_time and not PAUSE_FLAG:
            last_time = time.time()
            save_data(color_img, depth_img, IMG_ID)
            IMG_ID += 1
            cnt -= 1

        # key control
        if c == 27 or c == ord('q') or cnt==0:
            cv2.destroyAllWindows()
            exit()

        if c in [ord('r'), ord('R'), 13, 32]:
            save_data(color_img, depth_img, IMG_ID)
            IMG_ID += 1




