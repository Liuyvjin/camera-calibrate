import cv2
import numpy as np
from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
from pykinect2.mapper import depth_2_world, color_2_depth_space, depth_2_color_space, color_2_world

# kinect
class KinectCamera(object):
    def __init__(self ):
        kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color)
        self.camera = kinect
        self.color_shape = (kinect.color_frame_desc.Height, kinect.color_frame_desc.Width, 4)
        self.depth_shape = (kinect.depth_frame_desc.Height, kinect.depth_frame_desc.Width)
        print("color shape: ", self.color_shape)
        print("depth shape: ", self.depth_shape)

    def get_data(self):
        while True:
            if self.camera.has_new_depth_frame() and self.camera.has_new_color_frame():
                # 获取彩色图
                color_frame = self.camera.get_last_color_frame()
                color_img = color_frame.reshape(self.color_shape).astype(np.uint8)
                # 获取原始深度图
                # depth_frame = self.camera.get_last_depth_frame()
                # depth_img = depth_frame.reshape(self.depth_shape).astype(np.uint16)
                # 获取对齐到彩色图的深度
                depth_img = depth_2_color_space(self.camera, _DepthSpacePoint, self.camera._depth_frame_data, show=False, return_aligned_image=True)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.06), cv2.COLORMAP_JET)
                # 获取点云
                # pts = color_2_world(self.camera, self.camera._depth_frame_data, _CameraSpacePoint, as_array=True)
                return color_img, depth_img, depth_colormap