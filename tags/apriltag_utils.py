import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent)) # Project dir
import cv2
import numpy as np
from pupil_apriltags import Detector

CUR_DIR = Path(__file__).resolve().parent

class ApriltagUtils:
    """ Apriltag 工具 """
    def __init__(self, tag_size=2, families='tag36h11'):
        """
        :param get_img: 从相机获取图片的函数
        :param corner_file: 标定板角点坐标文件
        :param tag_size: tag 黑框的实际边长(cm)
        """
        self.at_detector = Detector(families=families,
                                nthreads=1,
                                quad_decimate=1.0,  # resolution / 1.0
                                quad_sigma=0.0,  # Gaussian blur, noisy images benefit from non-zero values
                                refine_edges=1,
                                decode_sharpening=0.25,
                                debug=0)
        self.tag_size = tag_size
        self.obj_p = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0],[-1, -1, 0]], dtype=np.float32) * tag_size / 2

    def find_corners(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.at_detector.detect(gray, estimate_tag_pose=False)

    def estimate_pose(self, img, mtx, dist, show=True) :
        """使用 OpenCV solvePnP 估计位姿

        :param img: 图像
        :param mtx: 相机内参矩阵
        :param dist: 相机畸变参数
        :param show: 在 img 中绘制角点
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tags = self.at_detector.detect(gray, estimate_tag_pose=False)
        for tag in tags:
            # 计算外参，即世界坐标系到相机坐标系的变换
            img_p = tag.corners[:, np.newaxis, :]
            ret, tag.pose_r, tag.pose_t = cv2.solvePnP(self.obj_p, img_p, mtx, dist)
            # 绘制角点，从 tag 的左下角点逆时针旋转
            if show:
                self.display_pose(img, tag, mtx, dist)
        return tags

    def estimate_pose_at(self, img, mtx, dist, show=True) :
        """使用 apriltag 估计位姿

        :param img: 图像
        :param mtx: 相机内参矩阵
        :param dist: 相机畸变参数
        :param show: 在 img 中绘制角点
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.undistort(gray, mtx, dist)  # 图像去畸变
        mtx_vec = [mtx[0,0], mtx[1,1], mtx[0,2], mtx[1,2]]
        tags = self.at_detector.detect(gray, estimate_tag_pose=True, camera_params=mtx_vec, tag_size=self.tag_size)
        for tag in tags:
            # 绘制角点，从 tag 的左下角点逆时针旋转
            tag.pose_r = cv2.Rodrigues(tag.pose_R)[0]
            if show:
                self.display_pose(img, tag, mtx, dist)
        return tags

    def display_pose(self, img, tag, mtx, dist):
        """ 可视化检测出的 tag 以及其位姿 """
        for i, c in enumerate(tag.corners):
            cv2.circle(img, tuple(c.astype(int)), 3, (0, 0, 255), 1)
            cv2.putText(img, str(i), tuple(c.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        # 将 3D 坐标系投影到图像平面
        axis_3d = np.float32([[0.5,0,0], [0,0.5,0], [0,0,-0.5]]) * self.tag_size  # 要绘制的 3D 点
        axis_2d = cv2.projectPoints(axis_3d, tag.pose_r, tag.pose_t, mtx, dist)[0].astype(int).squeeze()
        origin = tag.center.astype(int).reshape(-1)  # 注意第 4 个点才是原点
        cv2.line(img, origin, axis_2d[0], (255,0,0), 4) # B
        cv2.line(img, origin, axis_2d[1], (0,255,0), 4) # G
        cv2.line(img, origin, axis_2d[2], (0,0,255), 4) # R

if __name__ == '__main__':
    from cameras import Camera
    cam = Camera()
    cv2.namedWindow('tag', 0)

    # 导入相机参数
    with np.load(str(CUR_DIR / 'intrinsics.npz')) as file:  # 导入相机参数
        mtx, dist = [file[i] for i in ('mtx', 'dist')]

    # Apriltag
    apriltag = ApriltagUtils(tag_size=2, families='tag36h11')
    while True:
        img = cam.get_data() # 可视化 OpenCV 位姿估计结果
        tags = apriltag.estimate_pose(img, mtx, dist)
        if len(tags) > 0:
            print("pose_r: \n", tags[0].pose_r)
            print("pose_t: \n", tags[0].pose_t)

        cv2.imshow("tag", img)
        c = cv2.waitKey(10)
        if c == 27 or c == ord('q'):
            cv2.destroyAllWindows()
            exit()