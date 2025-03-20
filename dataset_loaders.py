import os
import cv2
import numpy as np

def load_inria_person_dataset(dataset_path, window_size=(64, 128)):
    pos_path = os.path.join(dataset_path, 'pos')
    neg_path = os.path.join(dataset_path, 'neg')
    annotations_path = os.path.join(dataset_path, 'annotations')
    
    X = []
    y = []
    annotations = []
    
    # 加载正样本和标注信息
    print('Loading positive samples and annotations...')
    pos_count = 0
    
    # 读取统一标注文件
    ann_file = os.path.join(dataset_path, 'annotations.txt')
    annotation_map = {}
    if os.path.exists(ann_file):
        with open(ann_file, 'r') as f:
            current_image = ''
            for line in f:
                line = line.strip()
                if line.startswith('Image:'):
                    current_image = line.split(': ')[-1].replace('.png', '')
                elif line.startswith('Bounding box'):
                    coords = line.split(': ')[-1].strip('()')
                    xmin, ymin, xmax, ymax = map(int, coords.replace('-', ',').split(','))
                    if current_image not in annotation_map:
                        annotation_map[current_image] = []
                    annotation_map[current_image].append((xmin, ymin, xmax, ymax))

    # 加载正样本
    for img_name in os.listdir(pos_path):
        if img_name.endswith('.png'):
            img_base = os.path.splitext(img_name)[0]
            img_path = os.path.join(pos_path, img_name)
            
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    # 获取该图片对应的所有标注框
                    boxes = annotation_map.get(img_base, [])
                    
                    # 调整图像大小
                    img = cv2.resize(img, window_size)
                    X.append(img)
                    y.append(1)
                    annotations.append(boxes)
                    pos_count += 1
    
    print(f'Loaded {pos_count} positive samples')

    # 加载负样本
    print('Loading negative samples...')
    neg_count = 0
    for img_name in os.listdir(neg_path):
        if img_name.endswith('.png'):
            img_path = os.path.join(neg_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                # 调整图像大小
                img = cv2.resize(img, window_size)
                X.append(img)
                y.append(0)
                annotations.append([])
                neg_count += 1
    
    print(f'Loaded {neg_count} negative samples')
    print(f'Total samples: {len(X)} (Positive: {pos_count}, Negative: {neg_count})')

    if pos_count == 0:
        print('Warning: No positive samples were loaded! Please check the dataset structure and annotations.txt file.')
    if neg_count == 0:
        print('Warning: No negative samples were loaded! Please check the dataset structure.')

    return np.array(X), np.array(y), annotations

def load_caltech_dataset(dataset_path, window_size=(64, 128)):
    pos_path = os.path.join(dataset_path, 'pos')
    neg_path = os.path.join(dataset_path, 'neg')
    
    X = []
    y = []
    annotations = []
    
    # 加载正样本
    print('Loading positive samples...')
    pos_count = 0
    for img_name in os.listdir(pos_path):
        if img_name.endswith('.jpg'):
            img_path = os.path.join(pos_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                # 调整图像大小
                img = cv2.resize(img, window_size)
                X.append(img)
                y.append(1)
                annotations.append([])
                pos_count += 1
    
    print(f'Loaded {pos_count} positive samples')

    # 加载负样本
    print('Loading negative samples...')
    neg_count = 0
    for img_name in os.listdir(neg_path):
        if img_name.endswith('.jpg'):
            img_path = os.path.join(neg_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                # 调整图像大小
                img = cv2.resize(img, window_size)
                X.append(img)
                y.append(0)
                annotations.append([])
                neg_count += 1
    
    print(f'Loaded {neg_count} negative samples')
    print(f'Total samples: {len(X)} (Positive: {pos_count}, Negative: {neg_count})')

    if pos_count == 0:
        print('Warning: No positive samples were loaded! Please check the dataset structure and annotations.txt file.')
    if neg_count == 0:
        print('Warning: No negative samples were loaded! Please check the dataset structure.')

    return np.array(X), np.array(y), annotations

def load_dataset(dataset_type, dataset_path):
    if dataset_type == 'inria':
        return load_inria_person_dataset(dataset_path)
    elif dataset_type == 'caltech':
        return load_caltech_dataset(dataset_path)
    else:
        raise ValueError(f'不支持的数据集类型: {dataset_type}')