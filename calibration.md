# Stereo Camera Calibration Guide

## Hardware Setup for OV9281 Pair

Mount both cameras on a rigid base (3D-printed or machined) with known baseline (distance between optical centers).

**Typical baseline for OV9281:** 100-120mm

```
  [LEFT CAM]  ----baseline----  [RIGHT CAM]
     OV9281         120mm         OV9281
       ↓                            ↓
   /dev/video0                  /dev/video2
```

---

## Step 1: Verify Device IDs

```bash
v4l2-ctl --list-devices
```

You'll see something like:
```
Arducam OV9281 (usb-xhci-hcd.1-1.1):
    /dev/video0
    /dev/video2
```

Update `config.yaml`:
```yaml
stereo_camera:
  left_device_id: "/dev/video0"
  right_device_id: "/dev/video2"
```

---

## Step 2: Intrinsic Calibration (Each Camera)

Use the ROS camera_calibration package or OpenCV:

```bash
# With ROS2
ros2 run camera_calibration cameracalibrator \
    --approximate 0.0 \
    --size 9x6 \
    --square 0.026 \
    image:=/stereo/left \
    camera:=/stereo_left
```

Or with a Python script using OpenCV (calibrate.py):

```python
import cv2
import numpy as np

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((9*6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 0.026

objpoints, imgpoints = [], []
# Collect 20+ images of checkerboard from different angles

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)
print(f"fx={mtx[0,0]:.1f}, fy={mtx[1,1]:.1f}, cx={mtx[0,2]:.1f}, cy={mtx[1,2]:.1f}")
```

Update `config.yaml` with the calibrated values:
```yaml
stereo_camera:
  fx: 450.0      # <-- replace with your calibrated fx
  cx: 640.0      # <-- replace with your calibrated cx
  cy: 400.0      # <-- replace with your calibrated cy
```

---

## Step 3: Stereo Extrinsic Calibration

This gives you the exact baseline and relative rotation between cameras:

```python
import cv2
import numpy as np

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

objp = np.zeros((9*6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 0.026

objpoints, imgpoints_l, imgpoints_r = [], [], []

# Capture synchronized stereo pairs of checkerboard (20+ pairs)
# Store them in two lists: left_images[], right_images[]

# Calibrate each camera individually
ret_l, mtx_l, dist_l, _, _ = cv2.calibrateCamera(
    objpoints, imgpoints_l, gray_l.shape[::-1], None, None
)
ret_r, mtx_r, dist_r, _, _ = cv2.calibrateCamera(
    objpoints, imgpoints_r, gray_r.shape[::-1], None, None
)

# Stereo calibration
ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_l, imgpoints_r,
    mtx_l, dist_l, mtx_r, dist_r,
    gray_l.shape[::-1],
    criteria=criteria,
    flags=cv2.CALIB_FIX_INTRINSIC,
)

baseline_mm = float(np.linalg.norm(T) * 1000)
print(f"Stereo baseline: {baseline_mm:.1f}mm")
print(f"Rotation matrix R:\n{R}")
print(f"Translation vector T: {T.flatten()}")
```

Update `config.yaml`:
```yaml
stereo_camera:
  baseline_mm: 120.0   # <-- replace with measured baseline in mm

stereo_depth:
  baseline_mm: 120.0   # must match above
  fx: 450.0             # from intrinsic calibration
```

---

## Step 4: Rectification (Optional — for aligned stereo)

To make the stereo pair perfectly horizontal (optional for SGBM):

```python
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    mtx_l, dist_l, mtx_r, dist_r,
    gray_l.shape[::-1], R, T,
    alpha=0,
)

# Save rectification maps
map_l_x, map_l_y = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, gray_l.shape[::-1], cv2.CV_32FC1)
map_r_x, map_r_y = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, gray_r.shape[::-1], cv2.CV_32FC1)

# Apply rectification before SGBM
left_rect = cv2.remap(left_gray, map_l_x, map_l_y, cv2.INTER_LINEAR)
right_rect = cv2.remap(right_gray, map_r_x, map_r_y, cv2.INTER_LINEAR)
```

---

## Quick Estimate (If No Checkerboard)

If you can't calibrate right now, use these estimates for OV9281 at 1280x800:

```yaml
stereo_camera:
  fx: 450.0
  cx: 640.0
  cy: 400.0
  baseline_mm: 120.0

stereo_depth:
  baseline_mm: 120.0
  fx: 450.0
```

---

## Testing Depth Estimation

Run standalone (before ROS):
```bash
cd autonomous-drone
python vision_pipeline.py
```

You should see:
- Left and right mono images
- Disparity map (brighter = closer)
- Depth map (color-coded: blue=far, red=close)

If depth is completely black or white, check:
1. Device IDs in config match `/dev/video*`
2. Baseline is correct (too small baseline = no detectable disparity)
3. Cameras are synchronized (both OV9281 on same USB controller)
