import cv2
import numpy as np
from sklearn.svm import LinearSVC
from hog_detector import HOGDetector

class LinearR2HOG(HOGDetector):
    def __init__(self, window_size=(64, 128), cell_size=(8, 8), block_size=(2, 2), block_sizes=[(2, 2), (3, 3)], nbins=9, sigma=0, norm_method='L2-Hys', threshold=0.5):
        super().__init__(window_size=window_size, nbins=nbins, sigma=sigma, norm_method=norm_method, threshold=threshold)
        self.window_size = window_size
        self.cell_size = cell_size
        self.block_size = block_size
        self.block_sizes = block_sizes
        self.nbins = nbins
        self.classifier = LinearSVC(random_state=42)

    def compute_gradient(self, img):
        # 计算x和y方向的梯度
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        # 计算梯度幅值和方向
        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = np.arctan2(gy, gx) * 180 / np.pi % 180
        return magnitude, orientation

    def compute_block_features(self, histograms, block_size):
        cell_rows, cell_cols, _ = histograms.shape
        features = []
        
        for row in range(cell_rows - block_size[1] + 1):
            for col in range(cell_cols - block_size[0] + 1):
                block = histograms[row:row+block_size[1],
                                 col:col+block_size[0], :]
                block_features = block.ravel()
                # 使用基类的归一化方法
                block_features = self.normalize_block(block_features)
                features.extend(block_features)
        
        return features

    def compute_hog_features(self, img):
        # 调整图像大小
        img = cv2.resize(img, self.window_size)
        # 转换为灰度图
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 计算梯度
        magnitude, orientation = self.compute_gradient(img)
        
        # 计算cell直方图
        cell_rows = self.window_size[1] // self.cell_size[1]
        cell_cols = self.window_size[0] // self.cell_size[0]
        histograms = np.zeros((cell_rows, cell_cols, self.nbins))
        
        for row in range(cell_rows):
            for col in range(cell_cols):
                cell_mag = magnitude[row*self.cell_size[1]:(row+1)*self.cell_size[1],
                                    col*self.cell_size[0]:(col+1)*self.cell_size[0]]
                cell_ori = orientation[row*self.cell_size[1]:(row+1)*self.cell_size[1],
                                      col*self.cell_size[0]:(col+1)*self.cell_size[0]]
                
                # 计算直方图
                hist_range = (0, 180)
                hist = np.histogram(cell_ori, bins=self.nbins, range=hist_range,
                                  weights=cell_mag)[0]
                histograms[row, col, :] = hist
        
        # 使用不同尺度的块进行特征提取
        features = []
        for block_size in self.block_sizes:
            block_features = self.compute_block_features(histograms, block_size)
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
                    decision_value = self.classifier.decision_function([features])[0]
                    if decision_value > self.threshold:
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