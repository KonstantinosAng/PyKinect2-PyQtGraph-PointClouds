# PyKinect2-PyQtGraph-PointClouds
Creating real-time dynamic Point Clouds using PyQtGraph and PyKinect2.

## Description
The PointCloud.py file contains the main class to produce dynamic Point Clouds using the [PyKinect2](https://github.com/Kinect/PyKinect2) and the [PyQtGraph](https://github.com/pyqtgraph/pyqtgraph) libraries.
The main file uses the numpy library that runs in C, thus it is fully optimized and can produce dynamic Point Clouds with up to 60+ frames, except for the point clouds produces by the RGB camera that run in 10+ frames.
The library can also be used to create PointCloud and save them as a .txt file containing the world point coordinates as: 
x, y, z
   .
   .
   . 
x, y, z

It can also be used to view .ply or .pcd point cloud files. Instructions on how to use the main file are written in the **Instructions** chapter.

## Requirements
Install all requirements from requirements.txt using the following command
```
pip install requirement
```
Full list of **all** requirements
```
* pyqtgraph==0.10.0
* numpy==1.18.2
* pykinect2==0.1.0
* opencv-python==4.2.0.34
* open3d-python==0.7.0.0
* time (already installed with Python)
* sys (already installed with Python)
* os (already installed with Python)
* ctypes (already installed with Python)
```

Another dependecy is the [mapper](https://github.com/KonstantinosAng/PyKinect2-Mapper-Functions) file that I created and handles the ICoordanteMapper functions. Download the file and place it in the same directory as the PointCloud.py file.

## Instructions

