import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent)) # Project dir
CUR_DIR = Path(__file__).resolve().parent
sys.path.append(CUR_DIR)
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QGridLayout, QSizePolicy
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtCore import QTimer, pyqtSignal
import pyqtgraph.opengl as gl

from apriltag_utils import ApriltagUtils
from cameras import Camera
from Calibrater import rtvec2transform
COLORS = (QColor(255,0,0), QColor(255,97,0), QColor(255,255,0), QColor(0,255,0), QColor(0,255,255), QColor(0,0,255), QColor(255,0,255))

class ApriltagUI(QWidget):
    # TagDetectedSignal = pyqtSignal(object)
    def __init__(self):
        super(ApriltagUI, self).__init__()
        # 相机和 at 检测器
        self.apriltag = ApriltagUtils(tag_size=2, families='tag36h11')
        self.cam = Camera()
        self.tag_maxval = 12 # 同时显示的最大 tag 数
        # 导入相机参数
        with np.load(str(CUR_DIR / 'intrinsics.npz')) as file:  # 导入相机参数
            self.mtx, self.dist = [file[i] for i in ('mtx', 'dist')]
        # ui
        self.ui_init()
        # 定时器，20ms 主循环
        self.timer = QTimer()
        self.timer.timeout.connect(self.main_loop)
        self.timer.start(0)
        # self.TagDetectedSignal.connect(self.draw_tags)

    def ui_init(self):
        self.width = 640 * 2 + 50
        self.height = 480 + 50  # 标题栏 30 px
        #根据显示器分辨率自动设置窗口大小
        self.setGeometry(100, 100, self.width, self.height)
        #设置窗口名称与标题
        self.setWindowTitle("Apriltag UI")

        self.label1 = QLabel(self)
        self.label1.setText('Camera')
        self.label1.setWordWrap(True)
        self.label1.setStyleSheet("QLabel{qproperty-alignment: AlignCenter; font-size:20px; font-weight:normal; font-family:宋体;}")
        self.label2 = QLabel(self)
        self.label2.setText('3D Pose')
        self.label2.setWordWrap(True)
        self.label2.setStyleSheet("QLabel{qproperty-alignment: AlignCenter;font-size:20px;font-weight:normal;font-family:宋体;}")

        self.cam_viewer = QLabel(self)
        self.cam_viewer.resize(640, 480)
        self.cam_viewer.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)  # 自由缩放
        self.gl_viewer = gl.GLViewWidget()          # 定义 opengl 3d 可视化窗口
        self.gl_viewer.resize(640, 480)
        self.gl_viewer.setCameraPosition(distance=30, elevation = 90, azimuth=90)
        self.draw_axis()

        self.tag_plots = []
        for i in range(self.tag_maxval):
            tag_plot = gl.GLLinePlotItem(width=2)
            self.tag_plots.append(tag_plot)
            self.gl_viewer.addItem(tag_plot)

        grid = QGridLayout()
        grid.addWidget(self.label1, 1, 1)
        grid.addWidget(self.label2, 1, 2)
        grid.addWidget(self.cam_viewer, 2, 1, 24, 1)
        grid.addWidget(self.gl_viewer, 2, 2, 24, 1)
        self.setLayout(grid)

    def draw_axis(self):
        self.gl_viewer.addItem(gl.GLGridItem())
        self.gl_viewer.addItem(gl.GLLinePlotItem(pos=[[0,0,0], [5,0,0]], color=QColor(0,0,255), width=5)) # x-blue
        self.gl_viewer.addItem(gl.GLTextItem(pos=[6,0,0], color=QColor(0,0,255), text='X'))
        self.gl_viewer.addItem(gl.GLLinePlotItem(pos=[[0,0,0], [0,5,0]], color=(0, 1, 0, 1), width=5)) # y-green
        self.gl_viewer.addItem(gl.GLTextItem(pos=[0,6,0], color=(125, 255, 125), text='Y'))
        self.gl_viewer.addItem(gl.GLLinePlotItem(pos=[[0,0,0], [0,0,5]], color=(1, 0, 0, 1), width=5)) # z-red
        self.gl_viewer.addItem(gl.GLTextItem(pos=[0,0,6], color=(255, 125, 125), text='Z'))

    def draw_tags(self, tags):
        for i, tag in zip(range(self.tag_maxval), tags):
            trans = rtvec2transform(tag.pose_r, tag.pose_t)
            obj_p = np.concatenate([self.apriltag.obj_p, np.ones((4, 1))], axis=1)
            pos = np.dot(obj_p, trans.T)[:, :3]  # tag 四个角点在相机坐标系下的坐标 (4, 3)
            pos = np.concatenate([pos, pos[:1], pos[2:3], pos[3:], pos[1:2]], axis=0)
            self.tag_plots[i].setData(pos = pos, color=COLORS[i%7])

        for i in range(len(tags), self.tag_maxval, 1):
            self.tag_plots[i].setData(color = (0, 0, 0, 0))  # 设为透明

    def main_loop(self):
        img = self.cam.get_data() # 可视化 OpenCV 位姿估计结果
        tags = self.apriltag.estimate_pose(img, self.mtx, self.dist)
        # self.TagDetectedSignal.emit(tags)
        self.draw_tags(tags)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame = QImage(img, img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888)
        picture = QPixmap.fromImage(frame).scaled(self.cam_viewer.width(), self.cam_viewer.height())
        self.cam_viewer.setPixmap(picture)

    def resizeEvent(self, event):
        w  = event.size().width()  * 640 // self.width
        h  = event.size().height() * 480 // self.height
        self.cam_viewer.resize(w, h)
        self.gl_viewer.resize(w, h)

    def closeEvent(self, event):
        exit(0)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ApriltagUI()
    win.show()
    sys.exit(app.exec_())