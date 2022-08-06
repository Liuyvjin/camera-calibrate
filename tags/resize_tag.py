import numpy as np
import cv2

if __name__ == '__main__':
    tag = cv2.imread("./tag36_11_00000.png")
    tag = cv2.resize(tag, np.array(tag.shape[:2]) * 30, interpolation=cv2.INTER_NEAREST )
    cv2.imwrite("./big_tag36_11_00000.png", tag)
    cv2.namedWindow("tag")
    cv2.imshow("tag", tag)
    cv2.waitKey(0)
