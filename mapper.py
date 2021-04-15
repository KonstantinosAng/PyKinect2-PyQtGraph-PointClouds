"""
Author: Konstantinos Angelopoulos
Date: 04/02/2020
All rights reserved.
Feel free to use and modify and if you like it give it a star.
"""


def subscribe_coordinate_mapping_changed(kinect):
    return kinect._mapper.SubscribeCoordinateMappingChanged()


def unsubscribe_coordinate_mapping_changed(kinect, waitableHandle_id):
    """
    The waitableHandle_id is returned by the subscribe_coordinate_mapping_changed function
    So use that function first to get the id and pass it to this function
    """
    return kinect._mapper.UnsubscribeCoordinateMappingChanged(waitableHandle_id)


def get_coordinate_mapping_changed_event_data(kinect, waitableHandle_id):
    """
        The waitableHandle_id is returned by the subscribe_coordinate_mapping_changed function
        So use that function first to get the id and pass it to this function
    """
    return kinect._mapper.GetCoordinateMappingChangedEventData(waitableHandle_id)


# Map Depth Space to Color Space (Image)
def depth_2_color_space(kinect, depth_space_point, depth_frame_data, show=False, return_aligned_image=False):
    """

    :param kinect: kinect class
    :param depth_space_point: _DepthSpacePoint from PyKinectV2
    :param depth_frame_data: kinect._depth_frame_data
    :param show: shows the aligned image
    :return: return the mapped color frame to depth frame
    """
    # Import here to optimize
    import numpy as np
    import ctypes
    import cv2
    # Map Color to Depth Space
    color2depth_points_type = depth_space_point * np.int(1920 * 1080)
    color2depth_points = ctypes.cast(color2depth_points_type(), ctypes.POINTER(depth_space_point))
    kinect._mapper.MapColorFrameToDepthSpace(ctypes.c_uint(512 * 424), depth_frame_data, ctypes.c_uint(1920 * 1080), color2depth_points)
    # Where color_point = [xcolor, ycolor]
    # color_x = color2depth_points[depth_point[1] * 1920 + color_point[0] - 1].x
    # color_y = color2depth_points[depth_point[1] * 1920 + color_point[0] - 1].y
    depthXYs = np.copy(np.ctypeslib.as_array(color2depth_points, shape=(kinect.color_frame_desc.Height*kinect.color_frame_desc.Width,)))  # Convert ctype pointer to array
    depthXYs = depthXYs.view(np.float32).reshape(depthXYs.shape + (-1,))  # Convert struct array to regular numpy array https://stackoverflow.com/questions/5957380/convert-structured-array-to-regular-numpy-array
    depthXYs += 0.5
    depthXYs = depthXYs.reshape(kinect.color_frame_desc.Height, kinect.color_frame_desc.Width, 2).astype(np.int)
    depthXs = np.clip(depthXYs[:, :, 0], 0, kinect.depth_frame_desc.Width - 1)
    depthYs = np.clip(depthXYs[:, :, 1], 0, kinect.depth_frame_desc.Height - 1)
    depth_frame = kinect.get_last_depth_frame()
    depth_img = depth_frame.reshape((kinect.depth_frame_desc.Height, kinect.depth_frame_desc.Width, 1)).astype(np.uint16)
    align_depth_img = np.zeros((1080, 1920, 4), dtype=np.uint16)
    align_depth_img[:, :] = depth_img[depthYs, depthXs, :]
    if show:
        cv2.imshow('Aligned Image', cv2.resize(cv2.flip(align_depth_img, 1), (int(1920 / 2.0), int(1080 / 2.0))))
        cv2.waitKey(3000)
    if return_aligned_image:
        return align_depth_img
    return depthXs, depthYs


# Map Color Space to Depth Space (Image)
def color_2_depth_space(kinect, color_space_point, depth_frame_data, show=False, return_aligned_image=False):
    """

    :param kinect: kinect class
    :param color_space_point: _ColorSpacePoint from PyKinectV2
    :param depth_frame_data: kinect._depth_frame_data
    :param show: shows aligned image with color and depth
    :return: mapped depth to color frame
    """
    import numpy as np
    import ctypes
    import cv2
    # Map Depth to Color Space
    depth2color_points_type = color_space_point * np.int(512 * 424)
    depth2color_points = ctypes.cast(depth2color_points_type(), ctypes.POINTER(color_space_point))
    kinect._mapper.MapDepthFrameToColorSpace(ctypes.c_uint(512 * 424), depth_frame_data, kinect._depth_frame_data_capacity, depth2color_points)
    # depth_x = depth2color_points[color_point[0] * 1920 + color_point[0] - 1].x
    # depth_y = depth2color_points[color_point[0] * 1920 + color_point[0] - 1].y
    colorXYs = np.copy(np.ctypeslib.as_array(depth2color_points, shape=(kinect.depth_frame_desc.Height * kinect.depth_frame_desc.Width,)))  # Convert ctype pointer to array
    colorXYs = colorXYs.view(np.float32).reshape(colorXYs.shape + (-1,))  # Convert struct array to regular numpy array https://stackoverflow.com/questions/5957380/convert-structured-array-to-regular-numpy-array
    colorXYs += 0.5
    colorXYs = colorXYs.reshape(kinect.depth_frame_desc.Height, kinect.depth_frame_desc.Width, 2).astype(np.int)
    colorXs = np.clip(colorXYs[:, :, 0], 0, kinect.color_frame_desc.Width - 1)
    colorYs = np.clip(colorXYs[:, :, 1], 0, kinect.color_frame_desc.Height - 1)
    color_frame = kinect.get_last_color_frame()
    color_img = color_frame.reshape((kinect.color_frame_desc.Height, kinect.color_frame_desc.Width, 4)).astype(np.uint8)
    align_color_img = np.zeros((424, 512, 4), dtype=np.uint8)
    align_color_img[:, :] = color_img[colorYs, colorXs, :]
    if show:
        cv2.imshow('img', cv2.flip(align_color_img, 1))
        cv2.waitKey(3000)
    if return_aligned_image:
        return align_color_img
    return colorXs, colorYs


# Map Color Points to Depth Points
def color_point_2_depth_point(kinect, depth_space_point, depth_frame_data, color_point):
    """

    :param kinect: kinect class
    :param depth_space_point: _DepthSpacePoint from PyKinectV2
    :param depth_frame_data: kinect._depth_frame_data
    :param color_point: color_point pixel location as [x, y]
    :return: depth point of color point
    """
    # Import here to optimize
    import numpy as np
    import ctypes
    # Map Color to Depth Space
    # Make sure that the kinect was able to obtain at least one color and depth frame, else the dept_x and depth_y values will go to infinity
    color2depth_points_type = depth_space_point * np.int(1920 * 1080)
    color2depth_points = ctypes.cast(color2depth_points_type(), ctypes.POINTER(depth_space_point))
    kinect._mapper.MapColorFrameToDepthSpace(ctypes.c_uint(512 * 424), depth_frame_data, ctypes.c_uint(1920 * 1080), color2depth_points)
    # Where color_point = [xcolor, ycolor]
    depth_x = color2depth_points[color_point[1] * 1920 + color_point[0] - 1].x
    depth_y = color2depth_points[color_point[1] * 1920 + color_point[0] - 1].y
    return [int(depth_x), int(depth_y)]


# Return depth of object given the depth map coordinates
def depth_space_2_world_depth(depth_map, x, y):
    """

    :param depth_map: kinect.get_last_depth_frame
    :param x: depth pixel x
    :param y: depth pixel y
    :return: depth z of object
    """
    if int(y) * 512 + int(x) <= 512 * 424:
        return float(depth_map[int(y) * 512 + int(x)])  # mm
    else:
        # If it exceeds return the last value to catch overflow
        return float(depth_map[512*424])


# Find the transformation from color to depth
def transform_color_2_depth(color_points, depth_points):
    """
    https://www.comp.nus.edu.sg/~cs4340/lecture/imorph.pdf
    Modules = import numpy as np
    TODO: Find 3 sets of (x, y) in color image and their (u, v) coordinates in depth image
    TODO: Use the sets to solve the following equations and find the coefficients
    Equation1: u =  a11*x + a12*y + a13
    Equation2: v =  a21*x + a22*y + a23
    :return: matrix with transformation coefficients [[a11, a12, a13], [a21, a22, a23]] as nparray

        [ depth_u ]   =   [a11 a12 a13]   *   [ color_x ]
        [ depth_v ]       [a21 a22 a23]   *   [ color_y ]
                                              [    1    ]
    """
    # Import library here to optimize
    import numpy as np
    import json
    ret = True
    matrix = []  # transformation matrix
    # Solve the first 3 equations to find coefficients for u coordinates
    color = np.array([[color_points[0][0], color_points[0][1], 1],
                      [color_points[1][0], color_points[1][1], 1],
                      [color_points[2][0], color_points[2][1], 1]])
    depth_u = np.array([depth_points[0][0], depth_points[1][0], depth_points[2][0]])
    depth_v = np.array([depth_points[0][1], depth_points[1][1], depth_points[2][1]])
    try:
        # Solve for u
        u_coeffs = np.linalg.solve(color, depth_u)
        # Solve for v
        v_coeffs = np.linalg.solve(color, depth_v)
        # Make matrix
        matrix = np.vstack([u_coeffs, v_coeffs])
        # Description for json file
        description = 'Transformation matrix to go from Color Coordinates to Depth Coordinates,\nwithout the need of MapColorFrameToDepthSpace from ICoordinateMapper.\nSee mapper.py for more information'
        # Write matrix to use in the main file
        with open('mapper/matrix.json', 'w', encoding='utf-8') as json_file:
            configs = {"Description": description, "Transformation Matrix": matrix.tolist()}
            json.dump(configs, json_file, separators=(',', ':'), sort_keys=True, indent=4)
    except Exception as e:
        print(f"[MAPPER]: Could not solve linear equations \n{e}")
        ret = False

    return matrix, ret


# Calculate pixel location from color to depth using only image resolutions
def xy2uv_with_res(x, y, color_width, color_height, depth_width, depth_height):
    """
    :return go from color pixel to depth pixel by ignoring distortion
    works better for center point but is not accurate for edge pixels
    """
    # Calculate pixel location assuming that images are the same and only the resolution changes
    u = (x / color_width) * depth_width
    v = (y / color_height) * depth_height
    return [u, v]


# Map Depth Frame to World Space
def depth_2_world(kinect, depth_frame_data, camera_space_point, as_array=False):
    """
    :param kinect: kinect class
     :param depth_frame_data: kinect._depth_frame_data
    :param camera_space_point: _CameraSpacePoint
    :param as_array: returns the data as a numpy array
    :return: returns the DepthFrame mapped to camera space
    """
    import numpy as np
    import ctypes
    depth2world_points_type = camera_space_point * np.int(512 * 424)
    depth2world_points = ctypes.cast(depth2world_points_type(), ctypes.POINTER(camera_space_point))
    kinect._mapper.MapDepthFrameToCameraSpace(ctypes.c_uint(512 * 424), depth_frame_data, ctypes.c_uint(512 * 424), depth2world_points)
    points = ctypes.cast(depth2world_points, ctypes.POINTER(ctypes.c_float))
    data = np.ctypeslib.as_array(points, shape=(424, 512, 3))
    if not as_array:
        return depth2world_points
    else:
        return data


# Map Color Frame to World Space
def color_2_world(kinect, depth_frame_data, camera_space_point, as_array=False):
    """
    :param kinect: Class for main file
    :param depth_frame_data: kinect._depth_frame_data
    :param camera_space_point: _CameraSpacePoint structure from PyKinectV2
    :param as_array: returns frame as numpy array
    :return: returns mapped color frame to camera space
    """
    import numpy as np
    import ctypes
    color2world_points_type = camera_space_point * np.int(1920 * 1080)
    color2world_points = ctypes.cast(color2world_points_type(), ctypes.POINTER(camera_space_point))
    kinect._mapper.MapColorFrameToCameraSpace(ctypes.c_uint(512 * 424), depth_frame_data, ctypes.c_uint(1920 * 1080), color2world_points)
    pf_csps = ctypes.cast(color2world_points, ctypes.POINTER(ctypes.c_float))
    data = np.ctypeslib.as_array(pf_csps, shape=(1080, 1920, 3))
    if not as_array:
        return color2world_points
    else:
        return data


# Map world/camera point to color space
def world_point_2_color(kinect, camera_space_point, point):
    """
    :arg: kinect class from main file
    :arg: _CameraSpacePoint structure from PyKinectV2
    :arg: world point [x, y, z] in meters
    :return: colorPoint = [u, v] pixel coordinates
    """
    import ctypes
    import numpy as np
    world_point_data_type = camera_space_point * np.int(1)
    world_point = ctypes.cast(world_point_data_type(), ctypes.POINTER(camera_space_point))
    world_point.contents.x = point[0]
    world_point.contents.y = point[1]
    world_point.contents.z = point[2]
    color_point = kinect._mapper.MapCameraPointToColorSpace(world_point.contents)
    return [color_point.x, color_point.y]


# Map world/camera point to depth space
def world_point_2_depth(kinect, camera_space_point, point):
    """
    :arg: kinect class from main file
    :arg: _CameraSpacePoint structure from PyKinectV2
    :arg: world point [x, y, z] in meters
    :return: depthPoint = [u, v] pixel coordinates
    """
    import ctypes
    import numpy as np
    world_point_data_type = camera_space_point * np.int(1)
    world_point = ctypes.cast(world_point_data_type(), ctypes.POINTER(camera_space_point))
    world_point.contents.x = point[0]
    world_point.contents.y = point[1]
    world_point.contents.z = point[2]
    depth_point = kinect._mapper.MapCameraPointToDepthSpace(world_point.contents)
    return [depth_point.x, depth_point.y]


# Map world/camera points to color space
def world_points_2_color(kinect, camera_space_point, points):
    """
    :arg: kinect class from main file
    :arg: _CameraSpacePoint structure from PyKinectV2
    :arg: world points [[x, y, z], [x, y, z], ..... , [x, y, z]] in meters
    :return: colorPoints = [[u, v], [u, v], ...., [u, v]] pixel coordinates
    """
    import ctypes
    import numpy as np
    world_point_data_type = camera_space_point * np.int(1)
    world_point = ctypes.cast(world_point_data_type(), ctypes.POINTER(camera_space_point))
    color_points = []
    for i in range(len(points)):
        world_point.contents.x = points[i, 0]
        world_point.contents.y = points[i, 1]
        world_point.contents.z = points[i, 2]
        color_point = kinect._mapper.MapCameraPointToColorSpace(world_point.contents)
        color_points.append([color_point.x, color_point.y])
    return color_points


# Map world/camera points to depth space
def world_points_2_depth(kinect, camera_space_point, points):
    """
    :arg: kinect class from main file
    :arg: _CameraSpacePoint structure from PyKinectV2
    :arg: world points [[x, y, z], [x, y, z], ..... , [x, y, z]] in meters
    :return: colorPoints = [[u, v], [u, v], ...., [u, v]] pixel coordinates
    """
    import ctypes
    import numpy as np
    world_point_data_type = camera_space_point * np.int(1)
    world_point = ctypes.cast(world_point_data_type(), ctypes.POINTER(camera_space_point))
    depth_points = []
    for i in range(len(points)):
        world_point.contents.x = points[i, 0]
        world_point.contents.y = points[i, 1]
        world_point.contents.z = points[i, 2]
        depth_point = kinect._mapper.MapCameraPointToDepthSpace(world_point.contents)
        depth_points.append([depth_point.x, depth_point.y])
    return depth_points


# Map Depth Points to Camera Space
def depth_points_2_world_points(kinect, depth_space_point, depth_points):
    """

    :param kinect: kinect class
    :param depth_space_point: _DepthSpacePoint
    :param depth_points: depth points as array [[x, y], [x, y], [x, y].... [x, y]]
    :return: return camera space points
    """
    import ctypes
    import numpy as np
    depth2world_point_type = depth_space_point * np.int(1)
    depth2world_point = ctypes.cast(depth2world_point_type(), ctypes.POINTER(depth_space_point))
    camera_points = np.ndarray(shape=(len(depth_points), 3), dtype=float)
    for i, point in enumerate(depth_points):
        depth2world_point.contents.x = point[0]
        depth2world_point.contents.y = point[1]
        world_point = kinect._mapper.MapDepthPointToCameraSpace(depth2world_point.contents, ctypes.c_ushort(512 * 424))
        camera_points[i] = [world_point.x, world_point.y, world_point.z]
    return camera_points  # meters


# Map depth points to world points faster than above method
def depth_points_2_camera_points(kinect, depth_space_point, camera_space_point, xys, as_array=False):
    """
    :param kinect: kinect class
    :param depth_space_point: _DepthSpacePoint
    :param camera_space_point: _CameraSpacePoint
    :return camera space points as camera_points[y*512 + x].x/y/z
    """
    import ctypes
    import numpy as np
    length_of_points = len(xys)
    depth_points_type = depth_space_point * np.int(length_of_points)
    depth_points = ctypes.cast(depth_points_type(), ctypes.POINTER(depth_space_point))
    camera_points_type = camera_space_point * np.int(length_of_points)
    camera_points = ctypes.cast(camera_points_type(), ctypes.POINTER(camera_space_point))
    depths = ctypes.POINTER(ctypes.c_ushort) * np.int(length_of_points)
    depths = ctypes.cast(depths(), ctypes.POINTER(ctypes.c_ushort))
    for i, point in enumerate(xys):
        depth_points[i].x = point[0]
        depth_points[i].y = point[1]
    kinect._mapper.MapDepthPointsToCameraSpace(ctypes.c_uint(length_of_points), depth_points, ctypes.c_uint(length_of_points), depths, ctypes.c_uint(length_of_points), camera_points)
    if as_array:
        camera_points = ctypes.cast(camera_points, ctypes.POINTER(ctypes.c_float))
        camera_points = np.ctypeslib.as_array(camera_points, shape=(length_of_points, 3))
        return camera_points
    return camera_points


# Map a depth point to world point
def depth_point_2_world_point(kinect, depth_space_point, depthPoint):
    """

    :param kinect: kinect class
    :param depth_space_point: _DepthSpacePoint from PyKinectV2
    :param depthPoint: depth point as array [x, y]
    :return: return the camera space point
    """
    # Import here for optimization
    import numpy as np
    import ctypes
    depth_point_data_type = depth_space_point * np.int(1)
    depth_point = ctypes.cast(depth_point_data_type(), ctypes.POINTER(depth_space_point))
    depth_point.contents.x = depthPoint[0]
    depth_point.contents.y = depthPoint[1]
    world_point = kinect._mapper.MapDepthPointToCameraSpace(depth_point.contents, ctypes.c_ushort(512*424))
    return [world_point.x, world_point.y, world_point.z]  # meters


# Map depth point to color point
def depth_point_2_color(kinect, depth_space_point, depthPoint):
    """
    :param kinect: kinect class
    :param depth_space_point: _DepthSpacePoint from PyKinectV2
    :param depthPoint: depth point as array [x, y]
    :return: return the mapped color point
    """
    import numpy as np
    import ctypes
    depth_point_type = depth_space_point * np.int(1)
    depth_point = ctypes.cast(depth_point_type(), ctypes.POINTER(depth_space_point))
    depth_point.contents.x = depthPoint[0]
    depth_point.contents.y = depthPoint[1]
    color_point = kinect._mapper.MapDepthPointToColorSpace(depth_point.contents, ctypes.c_ushort(512*424))
    return [color_point.x, color_point.y]


# Get Depth Frame to Camera Space Table
def depth_2_world_table(kinect, depth_space_point, as_array=False):
    """
    :param kinect: kinect instance
    :param depth_space_point: _DepthSpacePoint from PyKinectV2
    :param as_array: returns data as numpy array
    :return: return the mapped depth to camera space as frame
    """
    # Import here for optimization
    import numpy as np
    import ctypes
    table = depth_space_point * np.int(512 * 424)
    table = ctypes.cast(table(), ctypes.POINTER(ctypes.c_ulong))
    table = kinect._mapper.GetDepthFrameToCameraSpaceTable(table)
    """ Use table[0].x and table[0].y for the first pixel in kinect.get_last_depth_frame array
    """
    if as_array:
        """ Returns an array as table[0, 0] = x and table[0, 1] = y for the first pixel in depth frame
        """
        table = ctypes.cast(table, ctypes.POINTER(ctypes.c_float))
        table = np.ctypeslib.as_array(table, shape=(kinect.depth_frame_desc.Height * kinect.depth_frame_desc.Width, 2))
    return table


# Retrieve the depth camera intrinsics from the kinect's mapper
# and write them at: calibrate/IR/intrinsics_retrieved_from_kinect_mapper.json
def intrinsics(kinect, path='calibrate/IR/intrinsics_retrieved_from_kinect_mapper.json', write=False):
    """
    :param kinect: kinect instance
    :param path: path to save the intrinsics as a json file
    :param write: save or not save the intrinsics
    :return: returns the intrinsics matrix
    """
    import json
    intrinsics_matrix = kinect._mapper.GetDepthCameraIntrinsics()
    if write:
        with open(path, 'w', encoding='utf-8') as json_file:
            configs = {"FocalLengthX": intrinsics_matrix.FocalLengthX, "FocalLengthY": intrinsics_matrix.FocalLengthY,
                       "PrincipalPointX": intrinsics_matrix.PrincipalPointX, "PrincipalPointY": intrinsics_matrix.PrincipalPointY,
                       "RadialDistortionFourthOrder": intrinsics_matrix.RadialDistortionFourthOrder, "RadialDistortionSecondOrder": intrinsics_matrix.RadialDistortionSecondOrder,
                       "RadialDistortionSixthOrder": intrinsics_matrix.RadialDistortionSixthOrder}
            json.dump(configs, json_file, separators=(',', ':'), sort_keys=True, indent=4)
    return intrinsics_matrix


if __name__ == '__main__':
    """
        Example of some usages
    """
    from pykinect2 import PyKinectV2
    from pykinect2.PyKinectV2 import *
    from pykinect2 import PyKinectRuntime
    import cv2
    import numpy as np

    kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color)

    while True:
        if kinect.has_new_depth_frame():
            color_frame = kinect.get_last_color_frame()
            colorImage = color_frame.reshape((kinect.color_frame_desc.Height, kinect.color_frame_desc.Width, 4)).astype(np.uint8)
            colorImage = cv2.flip(colorImage, 1)
            cv2.imshow('Test Color View', cv2.resize(colorImage, (int(1920 / 2.5), int(1080 / 2.5))))
            depth_frame = kinect.get_last_depth_frame()
            depth_img = depth_frame.reshape((kinect.depth_frame_desc.Height, kinect.depth_frame_desc.Width)).astype(np.uint8)
            depth_img = cv2.flip(depth_img, 1)
            cv2.imshow('Test Depth View', depth_img)
            # print(color_point_2_depth_point(kinect, _DepthSpacePoint, kinect._depth_frame_data, [100, 100]))
            # print(depth_points_2_world_points(kinect, _DepthSpacePoint, [[100, 150], [200, 250]]))
            # print(intrinsics(kinect).FocalLengthX, intrinsics(kinect).FocalLengthY, intrinsics(kinect).PrincipalPointX, intrinsics(kinect).PrincipalPointY)
            # print(intrinsics(kinect).RadialDistortionFourthOrder, intrinsics(kinect).RadialDistortionSecondOrder, intrinsics(kinect).RadialDistortionSixthOrder)
            # print(world_point_2_depth(kinect, _CameraSpacePoint, [0.250, 0.325, 1]))
            # img = depth_2_color_space(kinect, _DepthSpacePoint, kinect._depth_frame_data, show=False, return_aligned_image=True)
            depth_2_color_space(kinect, _DepthSpacePoint, kinect._depth_frame_data, show=True)
            # img = color_2_depth_space(kinect, _ColorSpacePoint, kinect._depth_frame_data, show=True, return_aligned_image=True)

        # Quit using q
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cv2.destroyAllWindows()
