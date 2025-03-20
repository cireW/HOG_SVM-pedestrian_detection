import cv2
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt

class HOGDetector:
    def __init__(self, window_size=(64, 128), nbins=9, sigma=0.5, norm_method='L2-Hys', confidence_threshold=0.5):
        self.window_size = window_size
        self.nbins = nbins
        self.sigma = sigma
        self.norm_method = norm_method
        self.confidence_threshold = confidence_threshold
        self.classifier = LinearSVC(random_state=42)
        self.total_windows = 0
        self.false_positives = 0
        self.total_positives = 0
        self.missed_detections = 0

    def compute_gradient(self, img):
        # 如果需要，先进行高斯平滑
        if self.sigma > 0:
            img = cv2.GaussianBlur(img, (0, 0), self.sigma)
        # 计算x和y方向的梯度
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        # 计算梯度幅值和方向
        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = np.arctan2(gy, gx) * 180 / np.pi % 180
        return magnitude, orientation

    def normalize_block(self, block_features):
        if self.norm_method == 'L2-Hys':
            # L2-Hys归一化：先L2归一化，然后截断至0.2，再次L2归一化
            norm = np.linalg.norm(block_features) + 1e-6
            block_features /= norm
            block_features = np.clip(block_features, 0, 0.2)
            norm = np.linalg.norm(block_features) + 1e-6
            block_features /= norm
        elif self.norm_method == 'L2-norm':
            # L2归一化
            norm = np.sqrt(np.sum(block_features**2) + 1e-6)
            block_features = block_features / norm
        elif self.norm_method == 'L1-Sqrt':
            # L1-Sqrt归一化：先L1归一化，然后取平方根
            norm = np.sum(np.abs(block_features)) + 1e-6
            block_features = block_features / norm
            block_features = np.sqrt(np.abs(block_features))
        elif self.norm_method == 'L1-norm':
            # L1归一化
            norm = np.sum(np.abs(block_features)) + 1e-6
            block_features = block_features / norm
        return block_features

    def compute_hog_features(self, img):
        # 这是一个抽象方法，需要由子类实现
        raise NotImplementedError("子类必须实现compute_hog_features方法")

    def preprocess_image(self, img):
        # 调整图像大小
        img = cv2.resize(img, self.window_size)
        # 转换为灰度图
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def train(self, X, y):
        self.classifier = LinearSVC(random_state=42)

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
                    if decision_value > self.confidence_threshold:
                        detection_box = (int(x*scale), int(y*scale),
                                       int(min_size[0]*scale), int(min_size[1]*scale))
                        detections.append(detection_box)
                        
                        # 检查是否为误报
                        if ground_truth is not None:
                            is_true_positive = False
                            for gt_box in ground_truth:
                                # 计算IoU
                                intersection = self._compute_intersection(detection_box, gt_box)
                                union = self._compute_union(detection_box, gt_box)
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
    
    def _compute_intersection(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[0] + box1[2], box2[0] + box2[2])
        y2 = min(box1[1] + box1[3], box2[1] + box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0
        
        return (x2 - x1) * (y2 - y1)
    
    def _compute_union(self, box1, box2):
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        intersection = self._compute_intersection(box1, box2)
        return area1 + area2 - intersection

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