""" 对比 AprilTag 位姿估计和 OpenCV solvePnP 位姿估计的结果 """
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent)) # Project dir
import cv2
import numpy as np
from pupil_apriltags import Detector
from cameras import Camera

CUR_DIR = Path(__file__).resolve().parent


if __name__ == '__main__':
    cam = Camera()
    cv2.namedWindow('tag', 0)

    # 导入相机参数
    with np.load(str(CUR_DIR / 'intrinsics.npz')) as file:  # 导入相机参数
        mtx, dist = [file[i] for i in ('mtx', 'dist')]
    mtx_vec = [mtx[0,0], mtx[1,1], mtx[0,2], mtx[1,2]]
    # AprilTag 角点真实坐标
    tag_size = 2
    obj_point = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0],[-1, -1, 0]], dtype=np.float32) * tag_size / 2

    at_detector = Detector(families='tag36h11',
                        nthreads=1,
                        quad_decimate=1.0,  # resolution / 1.0
                        quad_sigma=0.0,  # Gaussian blur, noisy images benefit from non-zero values
                        refine_edges=1,
                        decode_sharpening=0.25,
                        debug=0)

    while True:
        img = cam.get_data() # 可视化 OpenCV 位姿估计结果
        img1 = img.copy() # 复制一份，可视化 AprilTag 位姿估计结果
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 使用 OpenCV 进行位姿估计
        tags = at_detector.detect(gray, estimate_tag_pose=False) # 如果手动定义 obj_point, tagsize 无影响
        for tag in tags:
            print("OpenCV Estimate: ")
            # 绘制角点，从 tag 的左下角点逆时针旋转
            for i, c in enumerate(tag.corners):
                cv2.circle(img, tuple(c.astype(int)), 3, (0, 0, 255), 1)
                cv2.putText(img, str(i), tuple(c.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.circle(img, tuple(tag.center.astype(int)), 4, (200,180,2), 4)
            # 计算外参，世界坐标系到相机坐标系的变换
            img_point = tag.corners[:, np.newaxis, :]
            ret, rvec, tvec = cv2.solvePnP(obj_point, img_point, mtx, dist)
            print(cv2.Rodrigues(rvec)[0])
            print(tvec)
            # 将 3D 坐标系投影到图像平面
            axis_3d = np.float32([[0.5,0,0], [0,0.5,0], [0,0,-0.5]])  # 要绘制的 3D 点
            axis_2d = cv2.projectPoints(axis_3d, rvec, tvec, mtx, dist)[0].astype(int).squeeze()
            origin = tag.center.astype(int).reshape(-1)  # 注意第 4 个点才是原点
            cv2.line(img, origin, axis_2d[0], (255,0,0), 5) # B
            cv2.line(img, origin, axis_2d[1], (0,255,0), 5) # G
            cv2.line(img, origin, axis_2d[2], (0,0,255), 5) # R

        # 使用 AprilTag 内置位姿估计
        gray_und = cv2.undistort(gray, mtx, dist)  # 图像去畸变
        tags1 = at_detector.detect(gray_und, estimate_tag_pose=True, camera_params=mtx_vec, tag_size=tag_size) # 位姿估计
        for tag in tags1:
            print("AprilTag Estimate: ")
            # 绘制检测到 tag 角点
            for i, c in enumerate(tag.corners):  # 从 tag 的左下角点逆时针旋转
                cv2.circle(img1, tuple(c.astype(int)), 3, (0, 0, 255), 1)
                cv2.putText(img1, str(i), tuple(c.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.circle(img1, tuple(tag.center.astype(int)), 4, (200,180,2), 4)
            # 输出外参
            print(tag.pose_R)
            print(tag.pose_t)
            # 将 3D 坐标系投影到图像平面
            axis_3d = np.float32([[0.5,0,0], [0,0.5,0], [0,0,-0.5]])  # 要绘制的 3D 点
            axis_2d = cv2.projectPoints(axis_3d, tag.pose_R, tag.pose_t, mtx, dist)[0].astype(int).squeeze()
            origin = tag.center.astype(int).reshape(-1)  # 注意第 4 个点才是原点
            cv2.line(img1, origin, axis_2d[0], (255,0,0), 5) # B
            cv2.line(img1, origin, axis_2d[1], (0,255,0), 5) # G
            cv2.line(img1, origin, axis_2d[2], (0,0,255), 5) # R


        cv2.putText(img, "solvePnP", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.putText(img1, "AprilTag", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.imshow("tag", np.hstack([img, img1]))
        c = cv2.waitKey(100)
        if c == 27 or c == ord('q'):
            cv2.destroyAllWindows()
            exit()
