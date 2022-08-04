import os
import cv2
import numpy as np
from pathlib import Path
import open3d as o3d
from cameras import KinectCamera as Camera

BASE_PATH = Path(__file__).resolve().parent
IMG_PATH = BASE_PATH / 'images'

# init record img id
def increment_path(path, name='color_', suffix='png'):
    if not path.exists():
        path.mkdir()
    for n in range(0, 9999):
        if not os.path.exists(f'{path}\{name}{n:03d}.{suffix}'):  #
            break
    return n
IMG_ID = increment_path(IMG_PATH, name = f'color_')

# save data
def save_data(color_img, depth_img, id):
    cv2.imwrite(f'{IMG_PATH}\color_{id:03d}.png', color_img)
    cv2.imwrite(f'{IMG_PATH}\depth_{id:03d}.png', depth_img)
    print(f'Saved at {IMG_PATH}\color_{id:03d}')

#  click mouse mid button to save img
def onMouse(event, x, y, flags, param):
    global IMG_ID
    if event == cv2.EVENT_MBUTTONDOWN:
        save_data(color_img, depth_img, IMG_ID)
        IMG_ID += 1


if __name__ == '__main__':
    cam = Camera()
    cv2.namedWindow('Kinect', 0)
    cv2.setMouseCallback('Kinect', onMouse)
    while True:
        color_img, depth_img, depth_colormap = cam.get_data()

        # Show images
        images = np.vstack([color_img[:,:,:3], depth_colormap])
        cv2.imshow('Kinect', images)
        c = cv2.waitKey(1)

        # open3d
        # ptsvis = pts.reshape(-1, 3)
        # ptsvis = pts[np.all(np.isfinite(ptsvis), -1)]
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(ptsvis)
        # o3d.visualization.draw_geometries([pcd])

        if c == 27 or c == ord('q'):
            cv2.destroyAllWindows()
            exit()
        if c in [ord('r'), ord('R'), 13, 32]:
            save_data(color_img, depth_img, IMG_ID)
            IMG_ID += 1

        # if c in [ord('f'), ord('F')]:
        #     cam.post_process ^= True
        #     print("Filter ", "ON" if cam.post_process else "OFF")
            # img = cv2.imread(depth_path, -1)



