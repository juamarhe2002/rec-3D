import cv2
import numpy as np
import glob, os, yaml
import matplotlib.pyplot as plt

# ───────────────────────── SETTINGS ────────────────────────── #
LEFT_IMG_GLOB  = "./fullbody/cameraL/calibration/*.jpg"     # pattern for left images
RIGHT_IMG_GLOB = "./fullbody/cameraR/calibration/*.jpg"    # pattern for right images
CHESSBOARD     = (9, 6)                   # inner corners  (cols, rows)
SQUARE_SIZE    = 1.9                      # cm

DEBUG_DRAW     = True                    # show detected patterns
# ────────────────────────────────────────────────────────────── #

###### 1. Grab image file lists ################################################
left_files  = sorted(glob.glob(LEFT_IMG_GLOB))
right_files = sorted(glob.glob(RIGHT_IMG_GLOB))
assert len(left_files) == len(right_files) and len(left_files) > 0, \
       "Left / right image counts differ or are zero."

###### 2. Prepare object-space corner grid #####################################
objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
objp[:, :2] = np.indices(CHESSBOARD).T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints     = []       # 3-D points in world coords
imgpoints_L   = []       # 2-D points in left  image
imgpoints_R   = []       # 2-D points in right image

criteria_subpix = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                   30, 0.001)
imgL_ref = cv2.imread(left_files[0])
imgR_ref = cv2.imread(right_files[0])
print("Image Left Shape: ", imgL_ref.shape[::-1])
print("Image Right Shape: ", imgR_ref.shape[::-1])
channels, width, height = imgL_ref.shape[::-1] # image size
img_size = (width, height)

print("[INFO] Detecting chessboard corners …")

for fL, fR in zip(left_files, right_files):
    imL, imR = cv2.imread(fL), cv2.imread(fR)

    grayL, grayR = cv2.cvtColor(imL, cv2.COLOR_BGR2GRAY), cv2.cvtColor(imR, cv2.COLOR_BGR2GRAY)

    okL, cornersL = cv2.findChessboardCorners(grayL, CHESSBOARD, None)
    okR, cornersR = cv2.findChessboardCorners(grayR, CHESSBOARD, None)
    if okL and okR:
        cv2.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria_subpix)
        cv2.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria_subpix)

        objpoints.append(objp)
        imgpoints_L.append(cornersL)
        imgpoints_R.append(cornersR)

    elif not okL and not okR:
        print(f"[ERROR] No chessboard corners detected for both images {fL} and {fR}.")
    elif not okL:
        print(f"[ERROR] No chessboard corners found for {fL}")
    elif not okR:
        print(f"[ERROR] No chessboard corners found for {fR}")

if DEBUG_DRAW:
    imL_corners = cv2.drawChessboardCorners(imL, CHESSBOARD, cornersL, okL)
    imR_corners = cv2.drawChessboardCorners(imR, CHESSBOARD, cornersR, okR)
    if imR.shape[::-1] != imgL_ref.shape[::-1]:
        imR_corners = cv2.resize(imR_corners, img_size)
    combined = np.hstack((imL_corners, imR_corners))
    plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    plt.title("Chessboard corners")
    plt.axis('off')
    plt.show()

print(f"[INFO] Kept {len(objpoints)} valid stereo pairs for calibration.")

###### 3. Calibrate cameras ############################################################
retval, K_L, D_L, rvecs_L, tvecs_L = cv2.calibrateCamera(objpoints, imgpoints_L, img_size, None, None)
assert retval, "Left camera calibration failed."

nK_L, roi_L = cv2.getOptimalNewCameraMatrix(
    K_L, D_L, img_size, 0, img_size
)

retval, K_R, D_R, rvecs_R, tvecs_R = cv2.calibrateCamera(objpoints, imgpoints_R, img_size, None, None)
assert retval, "Right camera calibration failed."

nK_R, roi_R = cv2.getOptimalNewCameraMatrix(
    K_R, D_R, img_size, 0, img_size
)

###### 4. Calibrate stereo ############################################################
stereoflags = cv2.CALIB_FIX_INTRINSIC
reProjErr, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_L, imgpoints_R,
    nK_L, D_L, nK_R, D_R, img_size,
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-6),
    flags=stereoflags
)
print(f"[INFO] pinhole stereo Reprojection Error = {reProjErr:.4f}")

R_L, R_R, P_L, P_R, Q, roi_L, roi_R = cv2.stereoRectify(
    nK_L, D_L, nK_R, D_R, img_size, R, T)

mapLx, mapLy = cv2.initUndistortRectifyMap(
    nK_L, D_L, R_L, P_L, img_size, cv2.CV_16SC2)

mapRx, mapRy = cv2.initUndistortRectifyMap(
    nK_R, D_R, R_R, P_R, img_size, cv2.CV_16SC2)

## Reprojection error.
def reprojection_error(objpoints, imgpoints, rvecs, tvecs, K, D):
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    return mean_error / len(objpoints)

total_error_left = reprojection_error(objpoints, imgpoints_L, rvecs_L, tvecs_L, nK_L, D_L)
print(f"Total reprojection error for left camera: {total_error_left:.4f}")

total_error_right = reprojection_error(objpoints, imgpoints_R, rvecs_R, tvecs_R, nK_R, D_R)
print(f"Total reprojection error for right camera: {total_error_right:.4f}")

###### 4. Save results #########################################################
def save_yaml(fname, data):
    with open(fname, "w") as f:
        yaml.dump({k: v.tolist() for k, v in data.items()}, f)

    print(f"[✓] Calibration file written: {fname}")

data = {
    "K_L":nK_L,
    "D_L":D_L,
    "K_R":nK_R,
    "D_R":D_R,
    "R":R,
    "T":T,
    "R_L":R_L,
    "roi_L": np.array(roi_L),
    "R_R":R_R,
    "roi_R": np.array(roi_R),
    "P_L":P_L,
    "P_R":P_R,
    "Q":Q
}
save_yaml("./calibration_params/calibration_data.yaml", data)

###### 5. Quick visual check ###########################################

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


if DEBUG_DRAW:
    imgL = cv2.imread(left_files[0])
    imgR = cv2.imread(right_files[0])

    imgR = match_image_shape_centered(imgL, imgR)

    # Undistort the images
    testL = cv2.remap(imgL, mapLx, mapLy, cv2.INTER_LINEAR)
    testR = cv2.remap(imgR, mapRx, mapRy, cv2.INTER_LINEAR)

    # Crop the images
    xl, yl, wl, hl = roi_L
    testL = testL[yl:yl+hl, xl:xl+wl]

    xr, yr, wr, hr = roi_R
    testR = testR[yr:yr+hr, xr:xr+wr]

    testR = match_image_shape_centered(testL, testR)

    vis = np.hstack((testL, testR))

    for i in range(0, vis.shape[0], 40):
        cv2.line(vis, (0, i), (vis.shape[1], i), (0, 255, 0), 1)

    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title("Rectified Stereo Pair")
    plt.axis('off')
    plt.show()
