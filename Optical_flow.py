import cv2 as cv
import numpy as np

class Optical_flow:
    def __init__(self, prev, curr, polylist):
     self.prev_frame = prev
     self.curr_frame = curr
     self.polylines = polylist

    def farne(self):
       flow = cv.calcOpticalFlowFarneback(self.prev_frame, self.curr_frame,
                                          None, pyr_scale=.5,
                                          winsiz=15, iterations=3,
                                          poly_n=5, poly_sigma=1.2,
                                          flags=0)
       return  flow
    
    def get_patch_flow(self, polyline):
        flow = Optical_flow(self.prev_frame, self.curr_frame).farne()
        dxs, dys = [], []

        for (x, y) in polyline:
          dx, dy = flow[int(y), int(x)]
          dxs.append(dx)
          dys.append(dy)
        
        return np.median(dxs), np.median(dys)
    
    def patch_update(self):
        for polyline in self.polylines:
            dx, dy = Optical_flow.get_patch_flow(self, polyline)
            # Updating centroid
            polyline[0] += dx
            polyline[1] += dy

            # Updating polyline outlines
            polyline = [(x+dx, y+dy) for (x,y) in polyline]
        return self.polylines