import cv2
import numpy as np
from sklearn.svm import LinearSVC
from hog_detector import HOGDetector

class LinearCHOG(HOGDetector):
    def __init__(self, window_size=(64, 128), cell_radius=8, block_size=(2, 2), nbins=9, sigma=0, norm_method='L2-Hys'):
        super().__init__(window_size=window_size, nbins=nbins, sigma=sigma, norm_method=norm_method)
        self.cell_radius = cell_radius
        self.block_size = block_size

    def compute_gradient(self, img):
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = np.arctan2(gy, gx) * 180 / np.pi % 180
        return magnitude, orientation

    def create_circular_mask(self, h, w, center, radius):
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        mask = dist_from_center <= radius
        return mask

    def compute_hog_features(self, img):
        img = cv2.resize(img, self.window_size)
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        magnitude, orientation = self.compute_gradient(img)
        
        # 计算圆形cell的中心点
        cell_spacing = self.cell_radius * 2
        cell_centers_x = np.arange(self.cell_radius, self.window_size[0], cell_spacing)
        cell_centers_y = np.arange(self.cell_radius, self.window_size[1], cell_spacing)
        cell_rows = len(cell_centers_y)
        cell_cols = len(cell_centers_x)
        
        histograms = np.zeros((cell_rows, cell_cols, self.nbins))
        
        for i, y in enumerate(cell_centers_y):
            for j, x in enumerate(cell_centers_x):
                mask = self.create_circular_mask(self.window_size[1], self.window_size[0],
                                               (x, y), self.cell_radius)
                cell_mag = magnitude[mask]
                cell_ori = orientation[mask]
                
                hist_range = (0, 180)
                hist = np.histogram(cell_ori, bins=self.nbins, range=hist_range,
                                  weights=cell_mag)[0]
                histograms[i, j, :] = hist
        
        # 块归一化
        features = []
        for row in range(cell_rows - self.block_size[1] + 1):
            for col in range(cell_cols - self.block_size[0] + 1):
                block = histograms[row:row+self.block_size[1],
                                 col:col+self.block_size[0], :]
                block_features = block.ravel()
                norm = np.sqrt(np.sum(block_features**2) + 1e-6)
                block_features = block_features / norm
                features.extend(block_features)
        
        return np.array(features)

    def train(self, X, y):
        self.classifier.fit(X, y)

    def decision_function(self, X):
        return self.classifier.decision_function(X)

    def predict(self, X):
        return self.classifier.predict(X)

    def detect(self, img, ground_truth=None, scale_factor=1.05, min_size=(64, 128)):
        detections = []
        height, width = img.shape[:2]
        scale = 1
        self.total_windows = 0
        self.false_positives = 0
        
        # 如果提供了ground truth，更新total_positives
        if ground_truth is not None:
            self.total_positives = len(ground_truth)
        
        while True:
            scaled_width = int(width / scale)
            scaled_height = int(height / scale)
            if scaled_width < min_size[0] or scaled_height < min_size[1]:
                break
                
            scaled_img = cv2.resize(img, (scaled_width, scaled_height))
            for y in range(0, scaled_height - min_size[1], min_size[1]//8):
                for x in range(0, scaled_width - min_size[0], min_size[0]//8):
                    window = scaled_img[y:y+min_size[1], x:x+min_size[0]]
                    if window.shape[:2] != min_size:
                        continue
                    self.total_windows += 1
                    features = self.compute_hog_features(window)
                    pred = self.classifier.predict([features])[0]
                    if pred == 1:
                        detection = (int(x*scale), int(y*scale),
                                   int(min_size[0]*scale), int(min_size[1]*scale))
                        detections.append(detection)
                        
                        # 检查是否为误报
                        if ground_truth is not None:
                            is_true_positive = False
                            for gt_box in ground_truth:
                                intersection = self._compute_intersection(detection, gt_box)
                                union = self._compute_union(detection, gt_box)
                                iou = intersection / union if union > 0 else 0
                                
                                if iou > 0.5:  # IoU阈值
                                    is_true_positive = True
                                    break
                            
                            if not is_true_positive:
                                self.false_positives += 1
            scale *= scale_factor
        
        # 更新missed_detections
        if ground_truth is not None:
            detected_gt = set()
            for detection in detections:
                for i, gt_box in enumerate(ground_truth):
                    intersection = self._compute_intersection(detection, gt_box)
                    union = self._compute_union(detection, gt_box)
                    iou = intersection / union if union > 0 else 0
                    if iou > 0.5:
                        detected_gt.add(i)
            
            self.missed_detections = self.total_positives - len(detected_gt)
        
        return detections
    def get_fppw(self):
        """获取(missing rate, fppw)对
        Returns:
            tuple: (missing rate, fppw)
                - missing rate: 漏检率，missed_detections / total_positives
                - fppw: 每窗口误报率，false_positives / total_windows
        """
        missing_rate = self.missed_detections / self.total_positives if self.total_positives > 0 else 0
        fppw = self.false_positives / self.total_windows if self.total_windows > 0 else 0
        return (missing_rate, fppw)