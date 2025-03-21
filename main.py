import os
import cv2
import numpy as np
from sklearn.metrics import classification_report
from hog_detector import HOGDetector
import matplotlib.pyplot as plt
from dataset_loaders import load_dataset
from sklearn.metrics import confusion_matrix
import pandas as pd
def main():
    import argparse
    from HOGs.linear_rhog import LinearRHOG
    from HOGs.linear_chog import LinearCHOG
    from HOGs.linear_echog import LinearECHOG
    from HOGs.kernel_rhog import KernelRHOG
    from HOGs.linear_r2hog import LinearR2HOG

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='HOG行人检测器')
    parser.add_argument('--method', type=str, required=True,
                      choices=['rhog', 'chog', 'echog', 'krhog', 'r2hog'],
                      help='选择HOG方法: rhog (Linear R-HOG), chog (Linear C-HOG), '
                           'echog (Linear EC-HOG), krhog (Kernel R-HOG), '
                           'r2hog (Linear R2-HOG)')
    parser.add_argument('--sigma', type=float, default=0,
                      help='梯度尺度sigma，默认为0')
    parser.add_argument('--nbins', type=int, default=9,
                      help='方向直方图bins数量，默认为9')
    parser.add_argument('--norm-method', type=str, default='L2-Hys',
                      choices=['L2-Hys', 'L2-norm', 'L1-Sqrt', 'L1-norm'],
                      help='归一化方法，默认为L2-Hys')
    parser.add_argument('--window-size', type=int, nargs=2, default=[64, 128],
                      metavar=('width', 'height'),
                      help='检测窗口大小，默认为64 128')
    parser.add_argument('--cell-size', type=int, nargs=2, default=[6, 6],
                      metavar=('width', 'height'),
                      help='Cell大小，默认为8 8')
    parser.add_argument('--block-size', type=int, nargs=2, default=[3, 3],
                      metavar=('width', 'height'),
                      help='Block大小（以cell为单位），默认为2 2')
    parser.add_argument('--dataset', type=str, default='inria',
                      choices=['inria', 'caltech'],
                      help='选择数据集: inria (INRIA Person), caltech (Caltech Pedestrian)')
    parser.add_argument('--threshold', type=float, default='1',
                      help='SVM分类器的置信阈值')
    args = parser.parse_args()

    # 根据选择初始化相应的HOG检测器
    detector_params = {
        'window_size': tuple(args.window_size),
        'cell_size': tuple(args.cell_size),
        'block_size': tuple(args.block_size),
        'nbins': args.nbins,
        'sigma': args.sigma,
        'norm_method': args.norm_method,
        'threshold': args.threshold
    }
    
    if args.method == 'rhog':
        detector = LinearRHOG(**detector_params)
    elif args.method == 'chog':
        detector = LinearCHOG(**detector_params)
    elif args.method == 'echog':
        detector = LinearECHOG(**detector_params)
    elif args.method == 'krhog':
        detector = KernelRHOG(**detector_params)
    else:  # r2hog
        detector = LinearR2HOG(**detector_params)
    
    # 加载数据集
    train_path = f'./datasets/{args.dataset.title()}/Train'
    test_path = f'./datasets/{args.dataset.title()}/Test'
    
    print(f'Loading {args.dataset.upper()} dataset...')
    X_train, y_train, train_annotations = load_dataset(args.dataset, train_path)
    X_test, y_test, test_annotations = load_dataset(args.dataset, test_path)
    
    
    # 提取HOG特征
    print('Extracting HOG features...')
    X_train_features = np.array([detector.compute_hog_features(img) for img in X_train])
    X_test_features = np.array([detector.compute_hog_features(img) for img in X_test])
    
    # 训练模型
    print('Training SVM classifier...')
    detector.train(X_train_features, y_train)
    
    # 评估模型
    print('Evaluating model...')
    # 获取预测的决策函数值（距离超平面的距离）
    y_scores = detector.decision_function(X_test_features)
    
    # 计算不同阈值下的Missing Rate和FPPW
    thresholds = np.linspace(np.min(y_scores), np.max(y_scores), 100)
    missing_rates = []
    fppw_rates = []
    
    # 计算测试集中的总正样本数量和总窗口数量
    total_positives = np.sum(y_test == 1)  # 使用y_test计算正样本数量
    total_windows = len(X_test)  # 每个图像视为一个检测窗口
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        
        # 统计漏检和误报
        missed_detections = 0
        false_positives = 0
        
        # 根据y_test计算漏检和误报
        for i, (pred, true_label) in enumerate(zip(y_pred, y_test)):
            if true_label == 1:  # 正样本图像
                if pred == 0:  # 预测为负样本
                    missed_detections += 1  # 漏检一个正样本
            else:  # 负样本图像
                if pred == 1:  # 预测为正样本
                    false_positives += 1
        
        # 计算Missing Rate和FPPW
        missing_rate = missed_detections / total_positives if total_positives > 0 else 0
        fppw = false_positives / total_windows if total_windows > 0 else 0
        
        missing_rates.append(missing_rate)
        fppw_rates.append(fppw)
    
    # 定义结果目录路径
    result_dir = './results'
    
    # 创建结果目录
    os.makedirs(result_dir, exist_ok=True)
    
    # 保存ROC曲线数据到CSV文件
    csv_file = os.path.join(result_dir, f'{args.method}_roc_data.csv')
    
    df = pd.DataFrame({
        'fppw': fppw_rates,
        'mr': missing_rates
    })
    df.to_csv(csv_file, index=False)
    
    
    # 测试检测效果
    # parser.add_argument('--test-image', type=str,
    #                   help='测试图片路径，用于可视化检测结果')
    
    # if args.test_image and os.path.exists(args.test_image):
    #     print('\nTesting detection on image...')
    #     test_img = cv2.imread(args.test_image)
    #     if test_img is not None:
    #         detections = detector.detect(test_img)
            
    #         # 可视化检测结果
    #         result_img = test_img.copy()
    #         for (x, y, w, h) in detections:
    #             cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
    #         plt.figure(figsize=(10, 6))
    #         plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    #         plt.title(f'Detection Results ({args.method})')
    #         plt.axis('off')
    #         plt.show()
    #     else:
    #         print(f'无法读取测试图片: {args.test_image}')

if __name__ == '__main__':
    main()