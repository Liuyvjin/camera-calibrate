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
        cv2.imshow("tag", img)
        c = cv2.waitKey(100)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        at_detector = Detector(families='tag36h11',
                            nthreads=1,
                            quad_decimate=1.0,  # resolution / 1.0
                            quad_sigma=0.0,  # Gaussian blur, noisy images benefit from non-zero values
                            refine_edges=1,
                            decode_sharpening=0.25,
                            debug=0)

        tags = at_detector.detect(gray, estimate_tag_pose=False, camera_params=None, tag_size=None)

        print(tags)