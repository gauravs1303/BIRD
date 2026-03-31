import ObstacleDetection 
import cv2 as cv
import numpy as np
class motion_map:
    def __init__(ObstacleDetection):
        pass

    def motion_mapping(prev_gray, gray, mag_thr=2.0):
        flow = cv.calcOpticleFlowFarneback(prev_gray, gray, None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        mag, ang = cv.cartToPolar(flow[..., 0], flow[...,1])
        moving = mag>mag_thr.astype(np.utint)*255
        return moving, mag, ang
    

#  # 🔹 Motion risk
#     if prev_gray is not None:
#         moving, mag, ang = motion_map(prev_gray, gray)
#         mag_small = cv.resize(mag, (pw, ph), interpolation=cv.INTER_AREA)
#         scores -= self.w_motion * mag_small  # penalize moving patches
