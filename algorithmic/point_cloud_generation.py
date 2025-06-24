import time

import cv2
import numpy as np
import yaml
import open3d as o3d
import os
import matplotlib.pyplot as plt
import pandas as pd

def match_image_shape_centered(left_img, right_img):
    """
    Modify the right image to match the shape of the left image by
    cropping or padding equally from all sides as needed.

    Args:
        left_img (np.ndarray): The reference image.
        right_img (np.ndarray): The image to modify.

    Returns:
        np.ndarray: Modified right image with the same shape as left image.
    """
    target_h, target_w = left_img.shape[:2]
    src_h, src_w = right_img.shape[:2]

    # Determine cropping or padding amounts
    diff_h = target_h - src_h
    diff_w = target_w - src_w

    # If cropping is needed
    if diff_h < 0:
        crop_top = (-diff_h) // 2
        crop_bottom = (-diff_h) - crop_top
    else:
        crop_top = crop_bottom = 0

    if diff_w < 0:
        crop_left = (-diff_w) // 2
        crop_right = (-diff_w) - crop_left
    else:
        crop_left = crop_right = 0

    # Crop the image if it's too large
    cropped = right_img[
        crop_top:src_h - crop_bottom if crop_bottom != 0 else None,
        crop_left:src_w - crop_right if crop_right != 0 else None
    ]

    # Recalculate dimensions after crop
    cropped_h, cropped_w = cropped.shape[:2]

    # Determine remaining padding amounts
    pad_h = target_h - cropped_h
    pad_w = target_w - cropped_w

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # Apply padding
    padded = cv2.copyMakeBorder(
        cropped,
        pad_top, pad_bottom,
        pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]  # Black padding
    )

    return padded


# --- Load stereo calibration data from YAML ---
def load_stereo_calibration(calibration_file):
    print(f"[INFO] Loading stereo calibration parameters from file: {calibration_file}")

    with open(calibration_file, 'r') as f:
        calib_data = yaml.safe_load(f)

    return calib_data

def load_Actor(calibration_file, camR, camL):

    df = pd.read_csv(calibration_file)

    K_L, Rvec_L, T_L, K_R, Rvec_R, T_R = None, None, None, None, None, None
    for i in range(len(df)):
        if camR != df['name'].iloc[i] and camL != df['name'].iloc[i]:
            continue

        if camR == df['name'].iloc[i]:
            K_R = np.array([
                [df['fx'].iloc[i], 0, df['px'].iloc[i]],
                [0, df['fy'].iloc[i], df['py'].iloc[i]],
                [0, 0, 1]
            ])
            Rvec_R = np.array([df['rx'].iloc[i], df['ry'].iloc[i], df['rz'].iloc[i]])
            T_R = np.array([df['tx'].iloc[i], df['ty'].iloc[i], df['tz'].iloc[i]])

        if camL == df['name'].iloc[i]:
            K_L = np.array([
                [df['fx'].iloc[i], 0, df['px'].iloc[i]],
                [0, df['fy'].iloc[i], df['py'].iloc[i]],
                [0, 0, 1]
            ])
            Rvec_L = np.array([df['rx'].iloc[i], df['ry'].iloc[i], df['rz'].iloc[i]])
            T_L = np.array([df['tx'].iloc[i], df['ty'].iloc[i], df['tz'].iloc[i]])

    if K_L is None or Rvec_L is None or T_L is None or T_R is None or K_R is None or Rvec_R is None or T_R is None:
        print("[ERROR] Failed to load camera parameters.")
        exit(1)

    return K_L, Rvec_L, T_L, K_R, Rvec_R, T_R

def downsample_image(image, reduce_factor):
    for i in range(0, reduce_factor):
        row, col = image.shape[0], image.shape[1]
        image = cv2.pyrDown(image, dstsize=(col//2, row//2))
    return image

def rectify_images(imgL, imgR, img_size, data_calib):
    print(f"[INFO] Rectifying images...")

    K_L = np.array(data_calib['K_L'])
    K_L = np.array(data_calib['K_L'])
    D_L = np.array(data_calib['D_L'])
    K_R = np.array(data_calib['K_R'])
    D_R = np.array(data_calib['D_R'])
    R_L = np.array(data_calib['R_L'])
    roi_L = np.array(data_calib['roi_L'])
    R_R = np.array(data_calib['R_R'])
    roi_R = np.array(data_calib['roi_R'])
    P_L = np.array(data_calib['P_L'])
    P_R = np.array(data_calib['P_R'])

    imgR = match_image_shape_centered(imgL, imgR)

    mapLx, mapLy = cv2.initUndistortRectifyMap(
        K_L, D_L, R_L, P_L, img_size, cv2.CV_16SC2)

    mapRx, mapRy = cv2.initUndistortRectifyMap(
        K_R, D_R, R_R, P_R, img_size, cv2.CV_16SC2)

    imgL_rect = cv2.remap(imgL, mapLx, mapLy, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    imgR_rect = cv2.remap(imgR, mapRx, mapRy, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    xl, yl, wl, hl = roi_L
    imgL_rect = imgL_rect[yl:yl + hl, xl:xl + wl]
    img_shape = imgL_rect.shape[::-1]

    xr, yr, wr, hr = roi_R
    imgL_rect = imgL_rect[yr:yr + hr, xr:xr + wr]

    imgR_rect = match_image_shape_centered(imgL_rect, imgR_rect)

    vis = np.hstack((imgL_rect, imgR_rect))
    for i in range(0, vis.shape[0], 40):
        cv2.line(vis, (0, i), (vis.shape[1], i), (0, 255, 0), 1)

    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.imshow(vis)
    plt.title("Rectified Stereo Pair")
    plt.axis("off")
    plt.show()
    return imgL_rect, imgR_rect, img_shape

# --- Compute disparity ---
def compute_disparity(imgL_rect, imgR_rect):
    print(f"[INFO] Computing disparity between images.")

    window_size = 5
    stereo = cv2.StereoSGBM.create(
        minDisparity=-1,
        numDisparities=32,
        blockSize=window_size,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=5,
        speckleWindowSize=10,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # WLS filter using the left matcher
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
    right_matcher = cv2.ximgproc.createRightMatcher(stereo)

    # Compute both disparities
    disp_left = stereo.compute(imgL_rect, imgR_rect).astype(np.float32) / 16.0
    disp_right = right_matcher.compute(imgL_rect, imgR_rect).astype(np.float32) / 16.0

    # Filter
    filtered_disp = wls_filter.filter(disp_left, imgL_rect, None, disp_right, right_view=imgR_rect)

    plt.imshow(filtered_disp, cmap='gray')
    plt.colorbar()
    plt.title("Filtered Disparity Map")
    plt.axis('off')
    plt.show()
    return filtered_disp

def save_point_cloud_ply(filename, points, colors):
    """
    Save a point cloud to a .ply file.

    Args:
        filename (str): Output file path.
        points (np.ndarray): Nx3 array of XYZ points.
        colors (np.ndarray, optional): Nx3 array of RGB colors (0-255). Must match length of points.
    """

    if colors is not None:
        assert colors.shape == points.shape, "Colors must match shape of points"
        has_color = True
    else:
        has_color = False

    with open(filename, 'w') as f:
        # Header
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(points)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        if has_color:
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
        f.write('end_header\n')

        # Data
        for i in range(len(points)):
            pt = points[i]
            if has_color:
                color = colors[i]
                f.write(f"{pt[0]} {pt[1]} {pt[2]} {int(color[0])} {int(color[1])} {int(color[2])}\n")
            else:
                f.write(f"{pt[0]} {pt[1]} {pt[2]}\n")

# --- Reproject to 3D and save point cloud ---
def generate_point_cloud(disparity, image, Q, output_path='output.ply'):
    print(f"[INFO] Generating point cloud...")
    points_3D = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=False)
    mask = np.isfinite(points_3D).all(axis=2) & (disparity > 0)
    points = points_3D[mask]
    colors = image[mask]

    save_point_cloud_ply(output_path, points, colors)
    print(f"[INFO] Saved point cloud to {output_path}")

def point_cloud_o3d(disparity, image, image_size, K, output_path='output.ply'):
    print(f"[INFO] Generating point cloud with Open3D...")

    depth_o3d = o3d.geometry.Image(disparity)
    image_o3d = o3d.geometry.Image(image)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d, depth_o3d,
                                                                    convert_rgb_to_intensity=False)

    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsic.set_intrinsics(
        image_size[0],
        image_size[1],
        K[0][0],
        K[1][1],
        K[0][2],
        K[1][2]
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)

    pcd.estimate_normals()
    pcd.orient_normals_to_align_with_direction()
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([pcd])

    o3d.io.write_point_cloud(output_path, pcd)
    print(f"[INFO] Saved point cloud to {output_path}")

# --- Main ---
if __name__ == "__main__":
    # Paths to inputs
    calib_file = "./calibration_params/calibration_data.yaml"
    # calib_Actors = "./imgs/calibration.csv"
    imgL_path = "./fullbody/cameraL/imgL_i1.jpg"
    imgR_path = "./fullbody/cameraR/imgR_i1.jpg"
    # imgL_path = "./imgs/Cam005_rgb000001.jpg"
    # imgR_path = "./imgs/Cam006_rgb000001.jpg"

    # Load data
    calib_data = load_stereo_calibration(calib_file)
    # K_L, Rvec_L, T_L, K_R, Rvec_R, T_R = load_Actor(calib_Actors, 'Cam005', 'Cam006')

    imgL = cv2.imread(imgL_path, cv2.IMREAD_COLOR_RGB)
    imgR = cv2.imread(imgR_path, cv2.IMREAD_COLOR_RGB)
    print("Image Left Shape: ", imgL.shape[::-1])
    print("Image Right Shape: ", imgR.shape[::-1])
    channels, width, height = imgL.shape[::-1]
    img_size = (width, height)

    if imgR.shape[::-1] != imgL.shape[::-1]:
        imgR = cv2.resize(imgR, img_size)

    vis = np.hstack((imgL, imgR))

    plt.imshow(vis)
    plt.title("Stereo Pair of Images")
    plt.axis('off')
    plt.show()

    grayL = cv2.imread(imgL_path, cv2.IMREAD_GRAYSCALE)

    # Rectify images.
    imgL_rect, imgR_rect, img_size = rectify_images(imgL, imgR, img_size, calib_data)

    # Compute disparity
    disp = compute_disparity(imgL_rect, imgR_rect)
    # disp = compute_disparity(imgL, imgR)

    # Save point cloud
    Q = np.array(calib_data['Q'])
    imgL_rect = cv2.cvtColor(imgL_rect, cv2.COLOR_BGR2RGB)
    generate_point_cloud(disp, imgL_rect, Q, output_path="pcd_i1.ply")

    K_L = np.array(calib_data['K_L'])
    point_cloud_o3d(disp, imgL_rect, img_size, K_L, output_path='pcd_i1_o3d.ply')
    # point_cloud_o3d(disp, imgL, img_size, K_L, output_path='pcd_Actor1_5_6.ply')
