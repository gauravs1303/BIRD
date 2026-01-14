import cv2 as cv
import numpy as np
from scipy import ndimage
from ultralytics import YOLO

class PatchScorer:
    def __init__(self, patch_size=32):
        self.patch_size = patch_size

        # weight
        self.w_edge = .50
        self.w_texture = .20
        self.w_blob = .15
        self.w_shadow = .15

    def preprocess(self, img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (5,5), 1.2)
        return gray
    
    def patchdim(self, img):
        h, w = img.shape[:2]
        ph = h // self.patch_size
        pw = w // self.patch_size
        return ph, pw
    
    #1 Edge Density = What fraction of pixels are edges?
    def edge_density(self, gray):
        gx = cv.Sobel(gray, cv.CV_32F, 1, 0)
        gy = cv.Sobel(gray, cv.CV_32F, 0, 1)
        mag = cv.magnitude(gx, gy)
        return mag

    #2 Texture Variance = Variance of pixel intensities in a local neighborhood
    def texture_variance(self, gray):
        mean = cv.blur(gray, (5,5))
        sqmean = cv.blur(gray.astype(np.float32)**2, (5,5))
        return sqmean - mean.astype(np.float32)**2


    # #3 Texture Energy = Sum of squared filter responses (Laplacian)
    # def texture_energy(self, gray):
    #     lap = cv.Laplacian(gray, cv.CV_32F)
    #     energy = cv.convertScaleAbs(lap)
    #     return energy
        
    #4 Blob Density = Response to LoG filter
    def blob_density(self, gray):
        log = cv.GaussianBlur(gray, (0,0), 2)
        log = cv.Laplacian(log, cv.CV_32F)
        return np.abs(log)
            
    #5 Shadow Map = Estimate shadow using HSV color space
    def shadow_map(self, img):
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        shadow = 1.0 - (hsv[:, :, 2]/255.0)
        return shadow
    
    # Normalize matrix to [0, 1]
    def normalize(self, m):
        m = m.astype(float)
        return (m-m.min()) / (m.max() - m.min() + 1e-6)

    # Compute all patch scores
    def compute_scores(self, img, obstacle_detector=None):
        gray = self.preprocess(img)
        ph, pw = self.patchdim(gray)

        edge_map = self.normalize(self.edge_density(gray))
        texture_var_map = self.normalize(self.texture_variance(gray))
        #texture_energy_map = self.normalize(self.texture_energy(gray))
        blob_map = self.normalize(self.blob_density(gray))
        shadow_map = self.normalize(self.shadow_map(img))

        # Downsampling
        edge_small = cv.resize(edge_map, (pw, ph), interpolation=cv.INTER_AREA)
        tex_small = cv.resize(texture_var_map, (pw, ph), interpolation=cv.INTER_AREA)
        shadow_small = cv.resize(shadow_map, (pw, ph), interpolation=cv.INTER_AREA)
        blob_small = cv.resize(blob_map, (pw, ph), interpolation=cv.INTER_AREA)

        # Vectorized scoring
        scores = (self.w_edge*(1-edge_small) + self.w_texture*(1-tex_small) + self.w_shadow*(1-shadow_small) + self.w_blob*(1-blob_small))
        if obstacle_detector is not None:
            obstacle_mask = obstacle_detector.obstacle_map(img)
            # Downsample obstacle mask to patch grid
            obstacle_mask_small = cv.resize(obstacle_mask, (pw, ph), interpolation=cv.INTER_AREA)
            scores[obstacle_mask_small > 0] = 0.0   # zero out scores where obstacles exist
        return scores
    
