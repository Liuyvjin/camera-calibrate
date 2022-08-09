import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent)) # Project dir
import cv2
import numpy as np
from pupil_apriltags import Detector
from cameras import Camera

CUR_PATH = str(Path(__file__).resolve().parent) + '\\'

class ApriltagCalibrater:
    def __init__(self, get_img, corner_file=CUR_PATH+'apriltag_corners.npz', tag_size=0.15):
        """
        :param get_img: 从相机获取图片的函数
        :param corner_file: 标定板角点坐标文件
        :param tag_size: tag 黑框的实际边长(m)
        """
        self.get_img = get_img
        # 准备角点真实坐标
        self.obj_p = np.load(corner_file)["corners"].astype(np.float32) * tag_size
        self.corner_num = self.obj_p.shape[0]
        self.obj_p = np.concatenate([self.obj_p, np.zeros((self.corner_num, 1), dtype=np.float32)], axis=1)

        # 相机像素
        img = get_img()
        self.h, self.w = img.shape[:2]
        print("Img shape: ({}, {})".format(self.w, self.h))
        # 迭代终止条件
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.001)

        # apriltag
        self.at_detector = Detector(families='tag36h11',
                                nthreads=1,
                                quad_decimate=1.0,  # resolution / 1.0
                                quad_sigma=0.0,  # Gaussian blur, noisy images benefit from non-zero values
                                refine_edges=1,
                                decode_sharpening=0.25,
                                debug=0)

    def calibrate(self, img_num, save=False):
        """交互采集 img_num 张图片, 标定 mtx 和 dist, 并进行重投影评估
        :param img_num: 拍摄图片数量
        :param save: 是否保存内参 mtx 和 dist
        """
        img_points = []
        obj_points = []
        cv2.namedWindow('calibrate', 0) # 实时显示
        print("Press Enter to capture image")
        cnt = 0
        while cnt < img_num:  # 共需要记录 self.img_num 组数据
            img = self.get_img()
            ret, corners = self.find_corners(img, show=True)
            cv2.imshow("calibrate", img)
            c = cv2.waitKey(5)
            if c == 13 and ret == True:  # 按下 enter 捕获当前图像角点
                img_points.append(corners)
                obj_points.append(self.obj_p)
                cnt += 1
                print("captured {} / {} images".format(cnt, img_num) )
            elif c == 27 or c == ord('q'):
                exit()
        cv2.destroyAllWindows()
        # 标定
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (self.w, self.h), None, None)
        print("mtx: \n", self.mtx)  # 内参矩阵
        print("dist: \n", self.dist)  # 畸变矩阵
        # 评估标定结果
        self.eval_reproj_error(img_points, rvecs, tvecs, self.mtx, self.dist)
        # 保存结果
        if save:
            np.savez(CUR_PATH+"intrinsics.npz", mtx=self.mtx, dist=self.dist)
        return self.mtx, self.dist

    def find_corners(self, img, show=True):
        """查找图像中的角点, 若找到返回 ret == True

        :param img: BGR 三通道彩色图像
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tags = self.at_detector.detect(gray, estimate_tag_pose=False, camera_params=None, tag_size=None)
        corners = np.zeros((self.corner_num, 1, 2), dtype=np.float32)
        for tag, i in zip(tags, range(0, len(tags)*4, 4)):  # 收集角点
            corners[i:i+4, 0] = tag.corners
        if show:
            cv2.drawChessboardCorners(img, (4, len(tags)), corners, True)  # 绘图
        if len(tags)*4 < self.corner_num:
            return False, None
        else:
            return True, corners

    def eval_reproj_error(self, img_points, rvecs, tvecs, mtx, dist):
        """计算重投影误差, 评估标定结果

        :param img_points: 角点像素坐标 (img_num, n*m, 2)
        :param rvecs: 旋转向量 (img_num, 3)
        :param tvecs: 位移向量 (img_num, 3)
        """
        mean_error = 0
        for i in range(len(rvecs)):
            reproj_p, _ = cv2.projectPoints(self.obj_p, rvecs[i], tvecs[i], mtx, dist)
            mean_error += cv2.norm(reproj_p, img_points[i], cv2.NORM_L2) / len(reproj_p)
        print( "mean reprojection error: {}".format(mean_error / len(img_points)) )

    def draw_axis(self, img):
        """实时绘制三维坐标
        :param img: BGR 彩色图像
        """
        axis_3d = np.float32([[0.5,0,0], [0,0.5,0], [0,0,-0.5]])  # 要绘制的 3D 点
        ret, corners = self.find_corners(img, show=False)
        if ret==True:
            ret, rvec, tvec = cv2.solvePnP(self.obj_p, corners, self.mtx, self.dist)
            # 将3D点投影到图像平面
            axis_2d = cv2.projectPoints(axis_3d, rvec, tvec, self.mtx, self.dist)[0].astype(int).squeeze()
            origin = corners[3].astype(int).ravel()  # 注意第 4 个点才是原点
            cv2.line(img, origin, axis_2d[0], (255,0,0), 5)  # B
            cv2.line(img, origin, axis_2d[1], (0,255,0), 5)  # G
            cv2.line(img, origin, axis_2d[2], (0,0,255), 5)  # R
        return img

if __name__ == '__main__':
    # from cameras import KinectCamera as Camera
    from cameras import Camera

    cam = Camera()
    get_data = lambda: cam.get_data()

    # 标定内参，并保存
    cali = ApriltagCalibrater(get_data)
    cali.calibrate(img_num=5, save=True)
    # 读入内参
    with np.load(CUR_PATH+'intrinsics.npz') as file:
        mtx, dist = [file[i] for i in ('mtx','dist')]
    print(mtx)
    print(dist)
    # 实时显示 3d 坐标
    cv2.namedWindow('3d_axis', cv2.WINDOW_AUTOSIZE)
    while True:
        img = get_data()
        img = cali.draw_axis(img)
        cv2.imshow('3d_axis', img)
        c = cv2.waitKey(1)
        if c == 27 or c == ord('q'):
            exit()
