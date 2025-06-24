# Code implementation of the reconstruction using algorithms

The script `stereo_calibrate.py` takes the calibration pictures on the `fullbody/cameraL/*.jpg` and `fullbody/cameraR/*.jpg` folders for calculating the calibration parameters of the stereo setup, and then saves them in the file `calibration_params/calibration_data.yaml`.

The script `point_cloud_generation.py` takes a specified pair of stereo images, then applies the rectification and undistortion according to the values saved on `calibration_params/calibration_data.yaml`, and finally generates a point cloud either with open3D or by manually writing the `.ply` file.
