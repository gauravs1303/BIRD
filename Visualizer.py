import cv2 as cv
import numpy as np

class Visualizer:
    def __init__(self, img, score, label=None):
        # Resize score to image size
        self.score_resized = cv.resize(score, (img.shape[1], img.shape[0]))
        # Create heatmap
        heat = cv.applyColorMap((self.score_resized*255).astype(np.uint8), cv.COLORMAP_JET)
        # Blend heatmap with original image
        self.blended = cv.addWeighted(img, 0.6, heat, 0.4, 0)
        # Thresholding to find safe zones
        max_val = np.max(self.score_resized)
        self.thre = 0.97 * max_val

    def countour(self):
        safe = np.uint8((self.score_resized >= self.thre)*255)
        cnts, _ = cv.findContours(safe, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        # Bounding counters
        if cnts:
            cmax = max(cnts, key=cv.contourArea)
            cmaxar = cv.contourArea(cmax)
            net = []
            for c in cnts:
                if cv.contourArea(c) < 0.5 * cmaxar:
                    continue
                epsilon = 0.01 * cv.arcLength(c, True)
                poly = cv.approxPolyDP(c, epsilon, True)
                polyline = cv.polylines(self.blended, [poly], True, (0,255,0), 2)
        
                # Compute centroid of polyline for text placement
                M = cv.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv.putText(self.blended, "Safe Zone", (cx, cy),
                            cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                
                # Compute mean and variance of the patch
                mask = np.zeros(self.score_resized.shape, dtype=np.uint8)
                cv.fillPoly(mask, [poly.astype(np.int32)], 1)
                values = self.score_resized[mask==1]

                net.append([cx, cy, polyline, values.mean(), values.var()])
        return net
        

    def show(self):
        # display = cv.resize(src=self.blended, dsize=(640, 4), interpolation=cv.INTER_LINEAR)
        cv.imshow("Landing Safety Heatmap", self.blended )
        cv.waitKey(1)
        # cv.destroyAllWindows()