# https://github.com/pupil-labs/apriltags
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent)) # Project dir
import cv2
import numpy as np
from pupil_apriltags import Detector
from cameras import Camera

if __name__ == '__main__':
    cam = Camera()
    cv2.namedWindow('tag', 0)
    while True:
        img = cam.get_data()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        at_detector = Detector(families='tag36h11',
                            nthreads=1,
                            quad_decimate=1.0,  # resolution / 1.0
                            quad_sigma=0.0,  # Gaussian blur, noisy images benefit from non-zero values
                            refine_edges=1,
                            decode_sharpening=0.25,
                            debug=0)

        tags = at_detector.detect(gray, estimate_tag_pose=False, camera_params=None, tag_size=None)

        id = 1
        for tag in tags:  # 绘制检测到 tag 角点
            for c in tag.corners:  # 从 tag 的左下角点逆时针旋转
                cv2.circle(img, tuple(c.astype(int)), 3, (0, 0, 255), 1)
                cv2.putText(img, str(id), tuple(c.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0), 1)
                id += 1
            cv2.circle(img, tuple(tag.center.astype(int)), 4, (200,180,2), 4)

        cv2.imshow("tag", img)
        c = cv2.waitKey(100)
        if c == 27 or c == ord('q'):
            cv2.destroyAllWindows()
            exit()

        # print(tags)