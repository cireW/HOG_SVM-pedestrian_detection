# HOG行人检测系统

本项目实现了基于HOG（Histogram of Oriented Gradients）特征的行人检测系统，包含多种HOG变体实现，可用于行人检测任务。

## 功能特点

- 支持多种HOG变体算法：
  - Linear R-HOG：标准的矩形HOG实现
  - Linear C-HOG：圆形HOG实现
  - Linear EC-HOG：椭圆形HOG实现
  - Kernel R-HOG：基于核函数的R-HOG实现
  - Linear R2-HOG：改进的矩形HOG实现
- 可配置的特征提取参数
- 基于SVM的行人分类器
- 支持多尺度检测
- 提供性能评估指标

## 安装依赖

```bash
pip install -r requirements.txt
```

## 数据集准备

本系统支持INRIA Person和Caltech Pedestrian两个数据集进行训练和测试。请按以下结构组织数据集：

### INRIA Person数据集
```
datasets/INRIAPerson/
├── Train/
│   ├── pos/  # 训练集正样本（包含行人的图像）
│   │   └── *.png
│   ├── neg/  # 训练集负样本（不包含行人的图像）
│   ├── └── *.png
│   └── annotations/ #bounding box
└── Test/
    ├── pos/  # 测试集正样本
    │   └── *.png
    ├── neg/  # 测试集负样本
    │   └── *.png
    └── annotations/ #bounding box

```

### Caltech Pedestrian数据集 (not support yet)
```
datasets/Caltech/
├── Train/
│   ├── pos/  # 训练集正样本（包含行人的图像）
│   │   └── *.jpg
│   └── neg/  # 训练集负样本（不包含行人的图像）
│       └── *.jpg
└── Test/
    ├── pos/  # 测试集正样本
    │   └── *.jpg
    └── neg/  # 测试集负样本
        └── *.jpg
```

## 使用方法

### 命令行参数
```bash
mkdir datasets
```

```bash
python main.py [参数]
```

必选参数：
- `--method`：选择HOG方法
  - `rhog`：Linear R-HOG
  - `chog`：Linear C-HOG
  - `echog`：Linear EC-HOG
  - `krhog`：Kernel R-HOG
  - `r2hog`：Linear R2-HOG

可选参数：
- `--sigma`：梯度计算的高斯平滑尺度（默认：0）
- `--nbins`：方向直方图的bins数量（默认：9）
- `--norm-method`：特征归一化方法
  - `L2-Hys`：L2范数归一化后截断（默认）
  - `L2-norm`：L2范数归一化
  - `L1-Sqrt`：L1范数归一化后取平方根
  - `L1-norm`：L1范数归一化
- `--window-size`：检测窗口大小，格式：宽 高（默认：64 128）

### 运行示例

1. 使用默认参数运行Linear R-HOG：
```bash
python main.py --method rhog
```

2. 使用自定义参数运行Linear C-HOG：
```bash
python main.py --method chog --sigma 1.0 --nbins 12 --norm-method L1-Sqrt --window-size 96 160
```

## 评估指标

系统提供以下评估指标：

- FPPW
- Missing rate

检测结果将以可视化方式展示，包括：
- 在测试图像上标注检测到的行人位置
- 显示检测边界框

## 注意事项

1. 在运行前请确保已正确设置数据集路径
2. 建议先使用小规模数据集进行测试
3. 不同的HOG变体可能需要不同的参数配置来获得最佳效果