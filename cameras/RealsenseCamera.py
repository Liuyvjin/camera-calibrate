import pyrealsense2 as rs
import numpy as np
import cv2

class RSCamera(object):
    def __init__(self, width=640, height=480):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
        self.pipeline_profile = self.pipeline.start(self.config)
        self.device = self.pipeline_profile.get_device()
        # 深度图后处理
        self.colorizer = rs.colorizer()
        self.post_process = True
        self.filters = [rs.spatial_filter(),
                        rs.temporal_filter(),]

        # 深度图对齐到彩色图
        self.align = rs.align(rs.stream.color)

        # cam init
        print('Camera init ...')
        for i in range(50):
            frames = self.pipeline.wait_for_frames()
        self.mtx = self.getIntrinsics()
        print('Camera init done.')

    def get_data(self):
        while True:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            # post-process
            if self.post_process:
                for f in self.filters:
                    depth_frame = f.process(depth_frame)

            if not depth_frame or not color_frame:
                continue

            depth_colormap = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            break
        return color_image, depth_image, depth_colormap

    def getIntrinsics(self):
        """ 获取内参 """
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        intrinsics = color_frame.get_profile().as_video_stream_profile().get_intrinsics()
        mtx = [intrinsics.width, intrinsics.height, intrinsics.ppx, intrinsics.ppy, intrinsics.fx, intrinsics.fy]
        camIntrinsics = np.array([[mtx[4],0,mtx[2]],
                                  [0,mtx[5],mtx[3]],
                                 [0,0,1.]])
        return camIntrinsics

    def __del__(self):
        self.pipeline.stop()


def vis_pc(pc):
    import open3d as o3d
    pc1 = o3d.geometry.PointCloud()
    pc1.points = o3d.utility.Vector3dVector(pc)
    o3d.visualization.draw_geometries([pc1])

if __name__ == "__main__":
    cam = RSCamera(width=640, height=480)
    depth_sensor = cam.pipeline_profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth scale is: ", depth_scale)

    while True:
        color, depth, depth_colormap = cam.get_data( )

        cv2.namedWindow('depth_colormap')
        cv2.imshow('depth_colormap', depth_colormap)
        cv2.namedWindow('color')
        cv2.imshow('color', color)
        c = cv2.waitKey(1)
        if c == 27: # 按下esc退出
            exit()
        if c == ord('f'):
            cam.post_process ^= True
            print("Post Process ", "On" if cam.post_process else "OFF")
