"""
Author: Konstantinos Angelopoulos
Date: 12/04/2020
All rights reserved.
Feel free to use and modify and if you like it give it a star.
"""

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import numpy as np
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
import mapper
import time
import cv2
from cv2 import WINDOW_FREERATIO
import sys
import os
import ctypes
import open3d as o3d


class Cloud:

    def __init__(self, file="", dynamic=False, color=False, depth=False, body=False, skeleton=False, simultaneously=False, color_overlay=False):
        """
        Initializes the point cloud
        file: The dir to the point cloud (either .txt, .ply or .pcd file)
        dynamic: Flag for displaying the pointcloud dynamically
        color: Flag for displaying the color pointcloud dynamically
        depth: Flag for displaying the depth pointcloud dynamically
        body: Flag for displaying the body index pointcloud dynamically
        skeleton: Flag for displaying the skeleton index pointcloud dynamically
        simultaneously: Flag for displaying more than one pointcloud dynamically (when one of the pointcloud is the skeleton you have to scroll out to see the pointcloud)
        :return None
        """
        # Initialize Kinect object
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color|PyKinectV2.FrameSourceTypes_Depth|PyKinectV2.FrameSourceTypes_Body|PyKinectV2.FrameSourceTypes_BodyIndex)
        self._body_index = None  # save body index image
        self._body_index_points = None  # save body index points
        self._cloud = False  # Flag to break loop when creating a pointcloud
        self._depth = None  # Store last depth frame
        self._color_frame = None  # store the last color frame
        self._red = 0  # Store red value from cv2 track bar
        self._green = 0  # Store green value from cv2 track bar
        self._blue = 0  # Store blue value from cv2 track bar
        self._size = 0.5  # Store value of point size from cv2 track bar
        self._opacity = 0  # store opacity value of colors from cv2 track bar
        self._dt = .0  # Store time value since kinect started from cv2 track bar
        self._skeleton_points = None  # store multiple skeleton points
        self._color_point_cloud = color  # Flag to show dynamic point cloud using the color frame
        self._depth_point_cloud = depth  # Flag to show dynamic point cloud using the depth frame
        self._simultaneously_point_cloud = simultaneously  # Flag for simultaneously showing the point clouds
        self._skeleton_point_cloud = skeleton  # Flag for showing the skeleton cloud
        self._dynamic = dynamic  # Flag for initializing a dynamic pointcloud
        self._cloud_file = file  # Store the file name
        self._body_index_cloud = body  # save body flag
        self._color_overlay = color_overlay  # flag to display the rgb image color up to the pointcloud
        self._dir_path = os.path.dirname(os.path.realpath(__file__))  # Store the absolute path of the file
        self._body_frame = None  # store body frame data
        self._joints = None  # save skeleton joints
        self._bodies_indexes = None  # save tracked skeleton indexes
        self._world_points = None  # Store world points
        self._color_point_cloud_points = None  # store color cloud points for simultaneously showing
        self._depth_point_cloud_points = None  # store depth cloud points for simultaneously showing
        self._body_point_cloud_points = None  # store body cloud points for simultaneously showing
        self._skeleton_point_cloud_points = None  # store skeleton cloud points for simultaneously showing
        self._simultaneously_point_cloud_points = None  # stack all the points
        self._skeleton_colors = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.float32)  # skeleton color pallet
        self._app = QtGui.QApplication([])  # Initialize app
        self._w = gl.GLViewWidget()  # Initialize view widget
        # Change view point
        self._w.orbit(225, -30)
        # self._w.pan(0, -2000, 0)  # make the camera fixed to a point
        # self._w.opts['viewport'] = (0, 0, 960, 540)
        self._w.showMaximized()  # show window maximized
        # self._w.setMaximumSize(960, 540)
        self._w.setWindowTitle('Kinect PointCloud')  # window title
        self._w.show()  # show app
        # self._g = gl.GLGridItem()  # adds a grid to the 3d space
        # self._g.setSize(x=1500, y=1500, z=1500)
        # self._w.addItem(self._g)
        self._scatter = None  # Store GL Scatter handler
        self._color = None  # Store color for each point
        self._t = None  # Store starting time for pointcloud
        self._start = True  # Flag for saving the main loop status
        self._start_gui = False  # Flag for stopping the main loop and exit when close
        self._dynamic_point_cloud = None  # Store the calculated point cloud points
        self._configurations = "configurations_input_window"  # cv2 window name for color and size
        self.create_track_bars()  # create track bars
        # check for multiple input flags or no input flags when using dynamic point cloud only
        if not self._simultaneously_point_cloud:
            if any([self._color_point_cloud and self._depth_point_cloud and self._body_index_cloud and self._skeleton_point_cloud,
                    self._dynamic and not self._color_point_cloud and not self._depth_point_cloud and not self._body_index_cloud and not self._skeleton_point_cloud,
                    self._color_point_cloud and self._depth_point_cloud,
                    self._color_point_cloud and self._body_index_cloud,
                    self._color_point_cloud and self._skeleton_point_cloud,
                    self._depth_point_cloud and self._body_index_cloud,
                    self._depth_point_cloud and self._skeleton_point_cloud,
                    self._body_index_cloud and self._skeleton_point_cloud]):
                # check for multiple flag inputs
                print('[CloudPoint] Too many arguments, choose color or depth pointcloud')
                print('Example 1 :\n pcl = Cloud(dynamic=True, color=True) \n pcl.visualize()')
                print('Example 2 :\n pcl = Cloud(dynamic=True, depth=True) \n pcl.visualize()')
                print('Example 3 :\n pcl = Cloud(dynamic=True, body=True) \n pcl.visualize()')
                print('Example 4 :\n pcl = Cloud(dynamic=True, skeleton=True) \n pcl.visualize()')
                sys.exit()
            else:
                if self._dynamic:
                    self.init()
                else:
                    if self._cloud_file != "":
                        # check if file is not a txt and is a pcd file
                        if self._cloud_file[-4:] == '.pcd' or self._cloud_file[-4:] == '.ply':
                            self.visualize_file()
                        elif self._cloud_file[-4:] == '.txt':
                            self.init()  # Initialize the GL GUI
                        else:
                            if '.' in self._cloud_file:
                                extension = '.' + self._cloud_file.split('.')[-1]
                                print('[CloudPoint] Not supported file extension ({})'.format(extension))
                                print('[CloudPoint] Only .txt, .pcd or .ply files are supported')
                            else:
                                print('[CloudPoint] Input has no valid file extension')
                            sys.exit()
        else:
            if self._dynamic:
                if  any([self._color_point_cloud and self._depth_point_cloud, self._color_point_cloud and self._body_index_cloud,
                         self._color_point_cloud and self._skeleton_point_cloud, self._depth_point_cloud and self._body_index_cloud,
                         self._depth_point_cloud and self._skeleton_point_cloud, self._body_index_cloud and self._skeleton_point_cloud]):
                    self.init()  # Initialize the GL GUI
                else:
                    # check for multiple flag inputs
                    print('[CloudPoint] Not Enough arguments, choose at least two methods of point clouds')
                    print('Example 1 :\n pcl = Cloud(dynamic=True, color=True, depth=True, simultaneously=True) \n pcl.visualize()')
                    print('Example 2 :\n pcl = Cloud(dynamic=True, color=True, body=True, simultaneously=True) \n pcl.visualize()')
                    print('Example 3 :\n pcl = Cloud(dynamic=True, color=True, skeleton=True, simultaneously=True) \n pcl.visualize()')
                    print('Example 4 :\n pcl = Cloud(dynamic=True, depth=True, body=True, simultaneously=True) \n pcl.visualize()')
                    print('Example 5 :\n pcl = Cloud(dynamic=True, depth=True, skeleton=True, simultaneously=True) \n pcl.visualize()')
                    print('Example 6 :\n pcl = Cloud(dynamic=True, body=True, skeleton=True, simultaneously=True) \n pcl.visualize()')
                    print('Example 7 :\n pcl = Cloud(dynamic=True, color=True, depth=True, body=True, skeleton=True, simultaneously=True) \n pcl.visualize()')
                    print('Example 8 :\n pcl = Cloud(dynamic=True, color=True, depth=True, skeleton=True, simultaneously=True) \n pcl.visualize()')
                    print('Example 9 :\n pcl = Cloud(dynamic=True, color=True, depth=True, body=True, simultaneously=True) \n pcl.visualize()')
                    print('Example 10 :\n pcl = Cloud(dynamic=True, color=True, skeleton=True, body=True, simultaneously=True) \n pcl.visualize()')
                    print('Example 11 :\n pcl = Cloud(dynamic=True, depth=True, skeleton=True, body=True, simultaneously=True) \n pcl.visualize()')
                    sys.exit()
            else:
                # check for multiple flag inputs
                print('[CloudPoint] Not Enough arguments, choose at least two methods of point clouds')
                print('Example 1 :\n pcl = Cloud(dynamic=True, color=True, depth=True, simultaneously=True) \n pcl.visualize()')
                print('Example 2 :\n pcl = Cloud(dynamic=True, color=True, body=True, simultaneously=True) \n pcl.visualize()')
                print('Example 3 :\n pcl = Cloud(dynamic=True, color=True, skeleton=True, simultaneously=True) \n pcl.visualize()')
                print('Example 4 :\n pcl = Cloud(dynamic=True, depth=True, body=True, simultaneously=True) \n pcl.visualize()')
                print('Example 5 :\n pcl = Cloud(dynamic=True, depth=True, skeleton=True, simultaneously=True) \n pcl.visualize()')
                print('Example 6 :\n pcl = Cloud(dynamic=True, body=True, skeleton=True, simultaneously=True) \n pcl.visualize()')
                print('Example 7 :\n pcl = Cloud(dynamic=True, color=True, depth=True, body=True, skeleton=True, simultaneously=True) \n pcl.visualize()')
                print('Example 8 :\n pcl = Cloud(dynamic=True, color=True, depth=True, skeleton=True, simultaneously=True) \n pcl.visualize()')
                print('Example 9 :\n pcl = Cloud(dynamic=True, color=True, depth=True, body=True, simultaneously=True) \n pcl.visualize()')
                print('Example 10 :\n pcl = Cloud(dynamic=True, color=True, skeleton=True, body=True, simultaneously=True) \n pcl.visualize()')
                print('Example 11 :\n pcl = Cloud(dynamic=True, depth=True, skeleton=True, body=True, simultaneously=True) \n pcl.visualize()')
                sys.exit()

        if self._cloud_file != "":
            # check if file is not a txt and is a pcd file
            if self._cloud_file[-4:] == '.pcd' or self._cloud_file[-4:] == '.ply':
                self.visualize_file()
            elif self._cloud_file[-4:] == '.txt':
                self.init()  # Initialize the GL GUI
            else:
                if '.' in self._cloud_file:
                    extension = '.' + self._cloud_file.split('.')[-1]
                    print('[CloudPoint] Not supported file extension ({})'.format(extension))
                    print('[CloudPoint] Only .txt, .pcd or .ply files are supported')
                else:
                    print('[CloudPoint] Input has no valid file extension')
                sys.exit()

    def create_track_bars(self):
        # Create window for track bars
        cv2.namedWindow(self._configurations, WINDOW_FREERATIO)
        cv2.createTrackbar("Size", self._configurations, 5, 350, self.nothing)
        cv2.createTrackbar("SkeletonSize", self._configurations, 0, 350, self.nothing)
        cv2.createTrackbar("Red", self._configurations, 255, 255, self.nothing)
        cv2.createTrackbar("Green", self._configurations, 255, 255, self.nothing)
        cv2.createTrackbar("Blue", self._configurations, 255, 255, self.nothing)
        cv2.createTrackbar("Opacity", self._configurations, 255, 255, self.nothing)
        cv2.createTrackbar("ColorOverlay", self._configurations, 0, 1, self.nothing)
        cv2.createTrackbar("Color Cloud", self._configurations, 0, 1, self.nothing)
        cv2.createTrackbar("Depth Cloud", self._configurations, 0, 1, self.nothing)
        cv2.createTrackbar("Body Cloud", self._configurations, 0, 1, self.nothing)
        cv2.createTrackbar("Skeleton Cloud", self._configurations, 0, 1, self.nothing)
        cv2.createTrackbar("Simultaneously", self._configurations, 0, 1, self.nothing)
        # update the positions
        if self._color_point_cloud:
            cv2.setTrackbarPos("Color Cloud", self._configurations, 1)
        if self._depth_point_cloud:
            cv2.setTrackbarPos("Depth Cloud", self._configurations, 1)
        if self._body_index_cloud:
            cv2.setTrackbarPos("Body Cloud", self._configurations, 1)
        if self._skeleton_point_cloud:
            cv2.setTrackbarPos("Skeleton Cloud", self._configurations, 1)
        if self._simultaneously_point_cloud:
            cv2.setTrackbarPos("Simultaneously", self._configurations, 1)
            cv2.setTrackbarPos("SkeletonSize", self._configurations, 20)
        if self._color_overlay:
            cv2.setTrackbarPos("ColorOverlay", self._configurations, 1)
            cv2.setTrackbarPos("Size", self._configurations, 30)

    def nothing(self, x):
        """
        For handling the callback from the cv2 track bar
        x: The callback returned for the cv2 track bar
        :return None
        """
        pass

    def create_points(self):
        """
        Check if the file exists and if not create the point cloud points and the file
        :return None
        """
        # Check if the file exists in the folder
        if not os.path.exists(os.path.join(self._dir_path, self._cloud_file)):
            if self._depth_point_cloud or self._color_point_cloud:
                t = time.time()  # starting time
                while not self._cloud:
                    # ----- Get Depth Frame
                    if self._kinect.has_new_depth_frame():
                        # store depth frame
                        self._depth = self._kinect.get_last_depth_frame()
                    # ----- Get Color Frame
                    if self._kinect.has_new_color_frame():
                        # store color frame
                        self._color_frame = self._kinect.get_last_color_frame()
                    # wait for kinect to grab at least one depth frame
                    if self._kinect.has_new_depth_frame() and self._color_frame is not None and self._dt > 6:

                        # use mapper to get world points
                        if self._depth_point_cloud:
                            world_points = mapper.depth_2_world(self._kinect, self._kinect._depth_frame_data, _CameraSpacePoint)
                            world_points = ctypes.cast(world_points, ctypes.POINTER(ctypes.c_float))
                            world_points = np.ctypeslib.as_array(world_points, shape=(self._kinect.depth_frame_desc.Height * self._kinect.depth_frame_desc.Width, 3))
                            world_points *= 1000  # transform to mm
                            self._dynamic_point_cloud = np.ndarray(shape=(len(world_points), 3), dtype=np.float32)
                            # transform to mm
                            self._dynamic_point_cloud[:, 0] = world_points[:, 0]
                            self._dynamic_point_cloud[:, 1] = world_points[:, 2]
                            self._dynamic_point_cloud[:, 2] = world_points[:, 1]

                            if self._cloud_file[-4:] == '.txt':
                                # remove zero depths
                                self._dynamic_point_cloud = self._dynamic_point_cloud[self._dynamic_point_cloud[:, 1] != 0]
                                self._dynamic_point_cloud = self._dynamic_point_cloud[np.all(self._dynamic_point_cloud != float('-inf'), axis=1)]

                            if self._cloud_file[-4:] == '.ply' or self._cloud_file[-4:] == '.pcd':
                                # update color for .ply file only
                                self._color = np.zeros((len(self._dynamic_point_cloud), 3), dtype=np.float32)
                                # map color to depth frame
                                Xs, Ys = mapper.color_2_depth_space(self._kinect, _ColorSpacePoint, self._kinect._depth_frame_data, show=False)
                                color_img = self._color_frame.reshape((self._kinect.color_frame_desc.Height, self._kinect.color_frame_desc.Width, 4)).astype(np.uint8)
                                # make align rgb/d image
                                align_color_img = np.zeros((self._kinect.depth_frame_desc.Height, self._kinect.depth_frame_desc.Width, 4), dtype=np.uint8)
                                align_color_img[:, :] = color_img[Ys, Xs, :]
                                align_color_img = align_color_img.reshape((self._kinect.depth_frame_desc.Height * self._kinect.depth_frame_desc.Width, 4)).astype(np.uint8)
                                align_color_img = align_color_img[:, :3:]  # remove the fourth opacity channel
                                align_color_img = align_color_img[..., ::-1]  # transform from bgr to rgb
                                self._color[:, 0] = align_color_img[:, 0]
                                self._color[:, 1] = align_color_img[:, 1]
                                self._color[:, 2] = align_color_img[:, 2]

                        if self._color_point_cloud:
                            # use mapper to get world points from color sensor
                            world_points = mapper.color_2_world(self._kinect, self._kinect._depth_frame_data, _CameraSpacePoint)
                            world_points = ctypes.cast(world_points, ctypes.POINTER(ctypes.c_float))
                            world_points = np.ctypeslib.as_array(world_points, shape=(self._kinect.color_frame_desc.Height * self._kinect.color_frame_desc.Width, 3))
                            world_points *= 1000  # transform to mm
                            # transform the point cloud to np (424*512, 3) array
                            self._dynamic_point_cloud = np.ndarray(shape=(len(world_points), 3), dtype=np.float32)
                            self._dynamic_point_cloud[:, 0] = world_points[:, 0]
                            self._dynamic_point_cloud[:, 1] = world_points[:, 2]
                            self._dynamic_point_cloud[:, 2] = world_points[:, 1]

                            if self._cloud_file[-4:] == '.txt':
                                # remove zeros from array
                                self._dynamic_point_cloud = self._dynamic_point_cloud[self._dynamic_point_cloud[:, 1] != 0]
                                self._dynamic_point_cloud = self._dynamic_point_cloud[np.all(self._dynamic_point_cloud != float('-inf'), axis=1)]

                            if self._cloud_file[-4:] == '.ply' or self._cloud_file[-4:] == '.pcd':
                                # update color for .ply file only
                                self._color = np.zeros((len(self._dynamic_point_cloud), 3), dtype=np.float32)
                                # get color image
                                color_img = self._color_frame.reshape((self._kinect.color_frame_desc.Height, self._kinect.color_frame_desc.Width, 4)).astype(np.uint8)
                                color_img = color_img.reshape((self._kinect.color_frame_desc.Height * self._kinect.color_frame_desc.Width, 4))
                                color_img = color_img[:, :3:]  # remove the fourth opacity channel
                                color_img = color_img[..., ::-1]  # transform from bgr to rgb
                                # update color with rgb color
                                self._color[:, 0] = color_img[:, 0]
                                self._color[:, 1] = color_img[:, 1]
                                self._color[:, 2] = color_img[:, 2]

                        # write points for txt file
                        if self._cloud_file[-4:] == '.txt':
                            row =''.join(','.join(str(point).strip('[]') for point in xyz) + '\n' for xyz in self._dynamic_point_cloud)
                            with open(os.path.join(self._dir_path, self._cloud_file), 'a') as txt_file:
                                txt_file.write(row)

                        self._cloud = True  # break loop
                    self._dt = time.time() - t  # running time
            else:
                print('[CloudPoint] No sensor flag checked')
                print('Example 1 :\n pcl = Cloud(file=filename, color=True) \n pcl.visualize()')
                print('Example 2 :\n pcl = Cloud(file=filename, depth=True) \n pcl.visualize()')
                sys.exit()

    def load_data(self):
        """
        Calculates the point cloud points only for one time for initialization purposes only
        :return None
        """
        # check for dynamic
        if not self._dynamic:
            # check if points are produced from pointcloud
            if self._dynamic_point_cloud is None:
                # Load data if file already existed
                with open(os.path.join(self._dir_path, self._cloud_file), 'r') as file:
                    # from string to float
                    data = [x for x in file.read().split('\n')]
                    data = [x.split(',') for x in data]
                # transform to array [:, 3]
                points = np.ndarray(shape=(len(data), 3), dtype=np.float32)
                for i, x in enumerate(data):
                    try:
                        points[i] = [float(x[0]), float(x[1]), float(x[2])]
                    except Exception as e:
                        pass
                # save points
                # self._dynamic_point_cloud = points[points[:, 1] != 0]  # its taken care in create_points function
                self._dynamic_point_cloud = points
            # save color for points
            self._color = np.zeros((self._dynamic_point_cloud.shape[0], 4), dtype=np.float32)
        else:
            # initialize zeros points just for initialization
            self._dynamic_point_cloud = np.ndarray(shape=(2, 3), dtype=np.float32)
            # Initialize color and plot the scatter points
            self._color = np.zeros((len(self._dynamic_point_cloud), 4), dtype=np.float32)
        self._color[:, :] = 1
        self._scatter = gl.GLScatterPlotItem(pos=self._dynamic_point_cloud, size=self._size, color=self._color)  # create first scatter points
        self._w.addItem(self._scatter)  # add items

    def update(self):
        """
        Update the position and color of the points inside the point cloud
        This functions is run in a thread loop and all code is optimized using
        numpy that runs in C to run faster.
        :return None
        """
        # Get track bar values
        self._size = cv2.getTrackbarPos("Size", self._configurations) / 10
        self._red = cv2.getTrackbarPos("Red", self._configurations)
        self._green = cv2.getTrackbarPos("Green", self._configurations)
        self._blue = cv2.getTrackbarPos("Blue", self._configurations)
        self._opacity = cv2.getTrackbarPos("Opacity", self._configurations)
        self._color_overlay = cv2.getTrackbarPos("ColorOverlay", self._configurations)
        # update the input track bar positions
        color = cv2.getTrackbarPos("Color Cloud", self._configurations)
        depth = cv2.getTrackbarPos("Depth Cloud", self._configurations)
        body = cv2.getTrackbarPos("Body Cloud", self._configurations)
        skeleton = cv2.getTrackbarPos("Skeleton Cloud", self._configurations)
        simultaneously = cv2.getTrackbarPos("Simultaneously", self._configurations)
        self._color_point_cloud = True if color == 1 else False
        self._simultaneously_point_cloud = True if simultaneously == 1 else False
        self._depth_point_cloud = True if depth == 1 else False
        self._body_index_cloud = True if body == 1 else False
        self._skeleton_point_cloud = True if skeleton == 1 else False

        # only for dynamic pointcloud
        if self._dynamic:

            # for color point cloud
            if self._color_point_cloud:
                # update the color points position
                self._world_points = mapper.color_2_world(self._kinect, self._kinect._depth_frame_data, _CameraSpacePoint, as_array=False)
                self._world_points = ctypes.cast(self._world_points, ctypes.POINTER(ctypes.c_float))
                self._world_points = np.ctypeslib.as_array(self._world_points, shape=(self._kinect.color_frame_desc.Height * self._kinect.color_frame_desc.Width, 3))
                # store points
                self._dynamic_point_cloud = np.ndarray(shape=(len(self._world_points), 3), dtype=np.float32)
                self._dynamic_point_cloud[:, 0] = self._world_points[:, 0] * 1000
                self._dynamic_point_cloud[:, 1] = self._world_points[:, 2] * 1000
                self._dynamic_point_cloud[:, 2] = self._world_points[:, 1] * 1000
                # remove zeros from array (it only has -inf instead of zeros like the depth frame)
                # self._dynamic_point_cloud = self._dynamic_point_cloud[self._dynamic_point_cloud[:, 1] != 0]
                # remove -inf (too slow)
                # self._dynamic_point_cloud = self._dynamic_point_cloud[np.all(self._dynamic_point_cloud != float('-inf'), axis=1)]
                # for simultaneously point clouds
                if self._simultaneously_point_cloud:
                    self._color_point_cloud_points = self._dynamic_point_cloud

            # for depth point cloud
            if self._depth_point_cloud:
                self._world_points = mapper.depth_2_world(self._kinect, self._kinect._depth_frame_data, _CameraSpacePoint)
                self._world_points = ctypes.cast(self._world_points, ctypes.POINTER(ctypes.c_float))
                self._world_points = np.ctypeslib.as_array(self._world_points, shape=(self._kinect.depth_frame_desc.Height * self._kinect.depth_frame_desc.Width, 3))
                # store points
                self._dynamic_point_cloud = np.ndarray(shape=(len(self._world_points), 3), dtype=np.float32)
                self._dynamic_point_cloud[:, 0] = self._world_points[:, 0] * 1000
                self._dynamic_point_cloud[:, 1] = self._world_points[:, 2] * 1000
                self._dynamic_point_cloud[:, 2] = self._world_points[:, 1] * 1000
                # remove -inf (too slow)
                # self._dynamic_point_cloud = self._dynamic_point_cloud[np.all(self._dynamic_point_cloud != float('-inf'), axis=1)]

                # simultaneously point cloud
                if self._simultaneously_point_cloud:
                    self._depth_point_cloud_points = self._dynamic_point_cloud

            # for body index point cloud
            if self._body_index_cloud:
                try:
                    # search for body index
                    self._body_index = self._kinect.get_last_body_index_frame().reshape((self._kinect.depth_frame_desc.Height, self._kinect.depth_frame_desc.Width)).astype(np.uint8)
                    # keep only the body index frame pixels
                    self._body_index_points = np.where(self._body_index != 255)
                    self._body_index_points = np.column_stack((self._body_index_points[0], self._body_index_points[1]))
                    self._body_index_points = self._body_index_points[self._body_index_points[:, 0] * 512 + self._body_index_points[:, 1] <= 512 * 424 - 1]
                    # calculate the world points from depth
                    world = mapper.depth_2_world_table(self._kinect, _DepthSpacePoint, as_array=False)
                    world = ctypes.cast(world, ctypes.POINTER(ctypes.c_float))
                    world = np.ctypeslib.as_array(world, shape=(self._kinect.depth_frame_desc.Height * self._kinect.depth_frame_desc.Width, 2))
                    #  get the depth frame
                    depth = self._kinect.get_last_depth_frame()
                    # calculate the world points for each body index frame pixel
                    self._dynamic_point_cloud = np.ndarray(shape=(len(self._body_index_points), 3), dtype=np.float32)
                    self._dynamic_point_cloud[:, 0] = world[self._body_index_points[:, 0] * 512 + self._body_index_points[:, 1]][:, 0] * 1000
                    self._dynamic_point_cloud[:, 1] = depth[self._body_index_points[:, 0] * 512 + self._body_index_points[:, 1]]
                    self._dynamic_point_cloud[:, 2] = world[self._body_index_points[:, 0] * 512 + self._body_index_points[:, 1]][:, 1] * 1000
                    # remove zero depth pixels
                    self._dynamic_point_cloud = self._dynamic_point_cloud[self._dynamic_point_cloud[:, 1] != 0]
                except:
                    # if no body frame is tracked then plot zeros
                    self._dynamic_point_cloud = np.ndarray(shape=(2, 3), dtype=np.float32)

                # simultaneously point cloud
                if self._simultaneously_point_cloud:
                    self._body_point_cloud_points = self._dynamic_point_cloud

            # for skeleton pointcloud
            if self._skeleton_point_cloud:
                try:
                    # search for tracked skeleton
                    self._bodies_indexes = []
                    # get body frame
                    self._body_frame = self._kinect.get_last_body_frame()
                    for i in range(0, self._kinect.max_body_count):
                        body = self._body_frame.bodies[i]
                        if not body.is_tracked:
                            continue
                        self._bodies_indexes.append(i)

                    # calculate the skeleton joints for each tracked skeleton
                    self._dynamic_point_cloud = np.ndarray(shape=(len(self._bodies_indexes) * 25, 3), dtype=np.float32)
                    for i, index in enumerate(self._bodies_indexes):
                        self._joints = self._body_frame.bodies[index].joints
                        self._dynamic_point_cloud[i*25:(i+1)*25, 0] = [joint.Position.x * 1000 for joint in self._joints[:25]]
                        self._dynamic_point_cloud[i*25:(i+1)*25, 1] = [joint.Position.z * 1000 for joint in self._joints[:25]]
                        self._dynamic_point_cloud[i*25:(i+1)*25, 2] = [joint.Position.y * 1000 for joint in self._joints[:25]]

                except:
                    # if no body is tracked then plot zeros
                    self._dynamic_point_cloud = np.ndarray(shape=(2, 3), dtype=np.float32)

                # simultaneously point cloud
                if self._simultaneously_point_cloud:
                    self._skeleton_point_cloud_points = self._dynamic_point_cloud

            # for simultaneously point cloud stack the point arrays
            if self._simultaneously_point_cloud:
                self._simultaneously_point_cloud_points = np.ndarray(shape=(1,3), dtype=np.float32)
                if self._color_point_cloud:
                    self._simultaneously_point_cloud_points = np.vstack((self._simultaneously_point_cloud_points, self._color_point_cloud_points))
                if self._depth_point_cloud:
                    depth_index_start = len(self._simultaneously_point_cloud_points) - 1
                    self._simultaneously_point_cloud_points = np.vstack((self._simultaneously_point_cloud_points, self._depth_point_cloud_points))
                    depth_index_end = len(self._simultaneously_point_cloud_points) - 1
                if self._body_index_cloud:
                    body_index_start = len(self._simultaneously_point_cloud_points) - 1
                    self._simultaneously_point_cloud_points = np.vstack((self._simultaneously_point_cloud_points, self._body_point_cloud_points))
                    body_index_end = len(self._simultaneously_point_cloud_points) - 1
                if self._skeleton_point_cloud:
                    self._simultaneously_point_cloud_points = np.vstack((self._simultaneously_point_cloud_points, self._skeleton_point_cloud_points))
                # remove the first initialized array
                self._dynamic_point_cloud = self._simultaneously_point_cloud_points[1:,:]

        # update the color and size of the points based on the track bars
        self._color = np.zeros((len(self._dynamic_point_cloud), 4), dtype=np.float32)
        self._color[:, 0] = self._red / 255
        self._color[:, 1] = self._green / 255
        self._color[:, 2] = self._blue / 255
        self._color[:, 3] = self._opacity / 255  # opacity

        # update color from rgb camera for each case
        if self._color_overlay:
            # update color from rgb camera when using the color img sensor
            if self._color_point_cloud:
                try:
                    # get color image
                    color_img = self._kinect.get_last_color_frame().reshape((self._kinect.color_frame_desc.Height, self._kinect.color_frame_desc.Width, 4)).astype(np.uint8)
                    color_img = np.divide(color_img, 255)  # standardize from 0 to 1
                    color_img = color_img.reshape((self._kinect.color_frame_desc.Height * self._kinect.color_frame_desc.Width, 4))
                    color_img = color_img[:, :3:]  # remove the fourth opacity channel
                    color_img = color_img[..., ::-1]  # transform from bgr to rgb
                    # update color with rgb color
                    self._color[:self._kinect.color_frame_desc.Height*self._kinect.color_frame_desc.Width, 0] = color_img[:, 0]
                    self._color[:self._kinect.color_frame_desc.Height*self._kinect.color_frame_desc.Width, 1] = color_img[:, 1]
                    self._color[:self._kinect.color_frame_desc.Height*self._kinect.color_frame_desc.Width, 2] = color_img[:, 2]
                except:
                    # handle exception during simultaneously where body is not yet tracked
                    pass

            # update color for the depth camera point cloud by mapping the rgb frame to the depth frame
            if self._depth_point_cloud:
                try:
                    # map color to depth frame
                    Xs, Ys = mapper.color_2_depth_space(self._kinect, _ColorSpacePoint, self._kinect._depth_frame_data, show=False)
                    color_img = self._kinect.get_last_color_frame().reshape((self._kinect.color_frame_desc.Height, self._kinect.color_frame_desc.Width, 4)).astype(np.uint8)
                    # make align rgb/d image
                    align_color_img = np.zeros((self._kinect.depth_frame_desc.Height, self._kinect.depth_frame_desc.Width, 4), dtype=np.uint8)
                    align_color_img[:, :] = color_img[Ys, Xs, :]
                    align_color_img = align_color_img.reshape((self._kinect.depth_frame_desc.Height*self._kinect.depth_frame_desc.Width, 4)).astype(np.uint8)
                    align_color_img = align_color_img[:, :3:]  # remove the fourth opacity channel
                    align_color_img = align_color_img[..., ::-1]  # transform from bgr to rgb
                    align_color_img = np.divide(align_color_img, 255)  # standardize from 0 to 1
                    # update color with rgb color
                    if self._simultaneously_point_cloud:
                        self._color[depth_index_start:depth_index_end, 0] = align_color_img[:, 0]
                        self._color[depth_index_start:depth_index_end, 1] = align_color_img[:, 1]
                        self._color[depth_index_start:depth_index_end, 2] = align_color_img[:, 2]
                    else:
                        self._color[:self._kinect.depth_frame_desc.Height*self._kinect.depth_frame_desc.Width, 0] = align_color_img[:, 0]
                        self._color[:self._kinect.depth_frame_desc.Height*self._kinect.depth_frame_desc.Width, 1] = align_color_img[:, 1]
                        self._color[:self._kinect.depth_frame_desc.Height*self._kinect.depth_frame_desc.Width, 2] = align_color_img[:, 2]
                except:
                    # handle exception during simultaneously where body is not yet tracked
                    pass

            # update color for the body index frame
            if self._body_index_cloud:
                try:
                    # if the depth point cloud is enabled remove these calculations
                    if not self._depth_point_cloud:
                        # map color to depth frame
                        Xs, Ys = mapper.color_2_depth_space(self._kinect, _ColorSpacePoint, self._kinect._depth_frame_data, show=False)
                        color_img = self._kinect.get_last_color_frame().reshape((self._kinect.color_frame_desc.Height, self._kinect.color_frame_desc.Width, 4)).astype(np.uint8)
                        # make align rgb/d image
                        align_color_img = np.zeros((self._kinect.depth_frame_desc.Height, self._kinect.depth_frame_desc.Width, 4), dtype=np.uint8)
                        align_color_img[:, :] = color_img[Ys, Xs, :]
                        align_color_img = align_color_img.reshape((self._kinect.depth_frame_desc.Height * self._kinect.depth_frame_desc.Width, 4)).astype(np.uint8)
                        align_color_img = align_color_img[:, :3:]  # remove the fourth opacity channel
                        align_color_img = align_color_img[..., ::-1]  # transform from bgr to rgb
                        align_color_img = np.divide(align_color_img, 255)  # standardize from 0 to 1
                    # remove zero depth points to match the array sizes as in the depth body index
                    self._body_index_points = self._body_index_points[depth[self._body_index_points[:, 0] * 512 + self._body_index_points[:, 1]] != 0]
                    # update color based on the rgb frame
                    if self._simultaneously_point_cloud:
                        self._color[body_index_start:body_index_end, 0] = align_color_img[self._body_index_points[:, 0] * 512 + self._body_index_points[:, 1]][:, 0]
                        self._color[body_index_start:body_index_end, 1] = align_color_img[self._body_index_points[:, 0] * 512 + self._body_index_points[:, 1]][:, 1]
                        self._color[body_index_start:body_index_end, 2] = align_color_img[self._body_index_points[:, 0] * 512 + self._body_index_points[:, 1]][:, 2]
                    else:
                        self._color[:self._kinect.depth_frame_desc.Height*self._kinect.depth_frame_desc.Width, 0] = align_color_img[self._body_index_points[:, 0] * 512 + self._body_index_points[:, 1]][:, 0]
                        self._color[:self._kinect.depth_frame_desc.Height*self._kinect.depth_frame_desc.Width, 1] = align_color_img[self._body_index_points[:, 0] * 512 + self._body_index_points[:, 1]][:, 1]
                        self._color[:self._kinect.depth_frame_desc.Height*self._kinect.depth_frame_desc.Width, 2] = align_color_img[self._body_index_points[:, 0] * 512 + self._body_index_points[:, 1]][:, 2]
                except:
                    # handle exceptions when no body is tracked
                    pass

        # update the skeleton color and size for simultaneously point cloud
        # for better visualization
        if self._skeleton_point_cloud and self._simultaneously_point_cloud:
            # make skeleton point bigger
            self._size = np.zeros(len(self._dynamic_point_cloud), dtype=np.float32)
            self._size[:] = cv2.getTrackbarPos("Size", self._configurations) / 10
            if len(self._bodies_indexes) > 0:
                self._size[-25*len(self._bodies_indexes):] = cv2.getTrackbarPos("SkeletonSize", self._configurations)
            # update the skeleton colors for each different skeleton tracked
            for i in range(len(self._bodies_indexes)):
                if i == 0:
                    self._color[-25:, 0] = self._skeleton_colors[i, 0]
                    self._color[-25:, 1] = self._skeleton_colors[i, 1]
                    self._color[-25:, 2] = self._skeleton_colors[i, 2]
                else:
                    self._color[-25*(i+1):-25*i, 0] = self._skeleton_colors[i, 0]
                    self._color[-25*(i+1):-25*i, 1] = self._skeleton_colors[i, 1]
                    self._color[-25*(i+1):-25*i, 2] = self._skeleton_colors[i, 2]

        # update the pyqtgraph cloud
        self._scatter.setData(pos=self._dynamic_point_cloud, color=self._color, size=self._size)

        if self._color_overlay:
            self._scatter.setGLOptions('opaque')  # enables depth and disables blending
        else:
            self._scatter.setGLOptions('additive')  # disables depth enables blending

    def init(self):
        """
        Initialize PyQTGraph and add the constructed points
        :return None
        """
        # check if the pointcloud is dynamically
        if not self._dynamic:
            self.create_points()
        self.load_data()  # load points for the first time
        self._t = QtCore.QTimer()  # initialize the Qui time
        self._t.timeout.connect(self.update)  # Initialize the update function
        self._t.start(10)  # import a delay

    def visualize(self):
        """
        Starting the visualization in pyqtgraph
        :return None
        """
        # start loop
        self._start = True
        while self._start:
            # check for interactive display and version
            if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
                # check to break loop
                if self._start_gui:
                    break
                # start app
                QtGui.QApplication.instance().exec_()
                self._start_gui = True
            else:
                self._start = False
        self._start = False
        cv2.destroyAllWindows()  # destroy track bar window and close application

    def visualize_file(self):
        """
        Handles the .pcd or .ply files visualization with Open3D
        :return None
        """
        import matplotlib.pyplot as plt
        self._w.close()  # close pyqtgraph window application
        QtGui.QApplication.quit()  # close pyqtgraph application
        cv2.destroyAllWindows()
        # Check if file exists
        if os.path.exists(os.path.join(self._dir_path, self._cloud_file)):
            vis = o3d.Visualizer()  # start visualizer
            vis.create_window(width=768, height=432)  # init window
            # add file geometry
            vis.add_geometry(o3d.read_point_cloud(os.path.join(self._dir_path, self._cloud_file)))
            opt = vis.get_render_option()  # get options
            opt.background_color = np.asarray([0, 0, 0])  # background to black
            view_control = vis.get_view_control()
            view_control.rotate(0, -360)
            vis.run()  # run visualization
            vis.destroy_window()  # destroy window after closing the point cloud
            sys.exit()  # exit the application
        else:
            # create and save file
            self.create_points()
            if self._cloud_file[-4:] == '.ply':
                self.export_to_ply()
            if self._cloud_file[-4:] == '.pcd':
                self.export_to_pcd()
            vis = o3d.Visualizer()  # start visualizer
            vis.create_window(width=768, height=432)  # init window
            # add file geometry
            vis.add_geometry(o3d.read_point_cloud(os.path.join(self._dir_path, self._cloud_file)))
            opt = vis.get_render_option()  # get options
            opt.background_color = np.asarray([0, 0, 0])  # background to black
            view_control = vis.get_view_control()
            view_control.rotate(0, -360)
            vis.run()  # run visualization
            vis.destroy_window()  # destroy window after closing the point cloud
            sys.exit()  # exit the application

    def export_to_ply(self):
        """
        Inspired by https://github.com/bponsler/kinectToPly
        Writes a kinect point cloud into a .ply file
        return None
        """
        # assert that the points have been created
        assert self._dynamic_point_cloud is not None, "Point Cloud has not been initialized"
        assert self._cloud_file != "", "Specify text filename"
        # stack data
        data = np.column_stack((self._dynamic_point_cloud, self._color))
        data = data[np.all(data != float('-inf'), axis=1)]  # remove -inf
        # header format of ply file
        header_lines = ["ply",
                        "format ascii 1.0",
                        "comment generated by: python",
                        "element vertex {}".format(int(len(data))),
                        "property float x",
                        "property float y",
                        "property float z",
                        "property uchar red",
                        "property uchar green",
                        "property uchar blue",
                        "end_header"]
        # convert to string
        data = '\n'.join('{} {} {} {} {} {}'.format('%.2f' % x[0], '%.2f' % x[1], '%.2f' % x[2], int(x[3]), int(x[4]), int(x[5])) for x in data)
        header = '\n'.join(line for line in header_lines) + '\n'
        # write file
        file = open(os.path.join(self._dir_path, self._cloud_file), 'w')
        file.write(header)
        file.write(data)
        file.close()

    def export_to_pcd(self):
        # assert that the points have been created
        assert self._dynamic_point_cloud is not None, "Point Cloud has not been initialized"
        assert self._cloud_file != "", "Specify text filename"
        # pack r/g/b to rgb
        rgb = np.asarray([[int(int(r_g_b[0]) << 16 | int(r_g_b[1]) << 8 | int(r_g_b[0]))] for r_g_b in self._color])
        # stack data
        data = np.column_stack((self._dynamic_point_cloud, rgb))
        data = data[np.all(data != float('-inf'), axis=1)]  # remove -inf
        # header format of pcd file
        header_lines = ["# .PCD v0.7 - Point Cloud Data file format",
                        "VERSION 0.7",
                        "FIELDS x y z rgb",
                        "SIZE 4 4 4 4",
                        "TYPE F F F U",
                        "COUNT 1 1 1 1",
                        "WIDTH {}".format(int(len(data))),
                        "HEIGHT 1",
                        "VIEWPOINT 0 0 0 1 0 0 0",
                        "POINTS {}".format(int(len(data))),
                        "DATA ascii"]
        # convert to string
        data = '\n'.join('{} {} {} {}'.format('%.2f' % x[0], '%.2f' % x[1], '%.2f' % x[2], int(x[3])) for x in data)
        header = '\n'.join(line for line in header_lines) + '\n'
        # write file
        file = open(os.path.join(self._dir_path, self._cloud_file), 'w')
        file.write(header)
        file.write(data)
        file.close()


if __name__ == "__main__":
    """
    For viewing a point cloud text file with: 
        x, y, z
        ....
        x, y, z
    (world point coordinates)
    If the file with the name does not exists it will create a point cloud with kinect
    and save it to that file.txt.
    It can also view .pcd and .ply files.
    """
    # pcl = Cloud(file='models/test_cloud_6.txt')
    # pcl.visualize()
    # pcd or ply files open with the Open3D library
    # pcl = Cloud(file='models/model.pcd')
    # pcl = Cloud(file='models/Car.ply')
    """
    If the files doesn't exist then you have to specify from which sensor camera you want 
    the pointcloud to be created and saved with that file name.
    """
    """ TXT files """
    # pcl = Cloud(file='models/test_cloud_7.txt', depth=True)
    # pcl.visualize()
    # pcl = Cloud(file='models/test_cloud_8.txt', color=True)
    # pcl.visualize()
    """ PLY files creation """
    # pcl = Cloud(file='models/test_cloud_10.ply', depth=True)
    # pcl = Cloud(file='models/test_cloud_10.ply', color=True)
    """ PCD files creation """
    # pcl = Cloud(file='models/test_cloud_10.pcd', depth=True)
    # pcl = Cloud(file='models/test_cloud_10.pcd', color=True)
    """
    For dynamically creating the PointCloud and viewing the PointCloud.
    """
    # rgb camera
    # pcl = Cloud(dynamic=True, color=True, color_overlay=False)
    # pcl.visualize()
    # depth camera
    # pcl = Cloud(dynamic=True, depth=True, color_overlay=True)
    # pcl.visualize()
    # body index
    # pcl = Cloud(dynamic=True, body=True, color_overlay=True)
    # pcl.visualize()
    # skeleton cloud
    #  pcl = Cloud(dynamic=True, skeleton=True, color_overlay=False)
    # pcl.visualize()
    """
    # You can also visualize the clouds simultaneously in any order, and apply the rgb frame color on top of them.s
    """
    # pcl = Cloud(dynamic=True, simultaneously=True, color=True, depth=True, body=False, skeleton=False, color_overlay=True)
    # pcl.visualize()
    # pcl = Cloud(dynamic=True, simultaneously=True, depth=True, color=True, body=True, skeleton=True, color_overlay=False)
    # pcl.visualize()
    # pcl = Cloud(dynamic=True, simultaneously=True, depth=True, color=False, body=True, skeleton=False, color_overlay=False)
    # pcl.visualize()
    pcl = Cloud(dynamic=True, simultaneously=True, depth=True, color=False, body=False, skeleton=True, color_overlay=True)
    pcl.visualize()
