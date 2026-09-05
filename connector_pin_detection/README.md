# 基于视觉的连接器引脚间距检测判定系统

## 项目简介

本项目设计一款基于机器视觉的电子连接器引脚间距检测判定系统，结合图像处理技术与尺寸测算算法，实现各类电子连接器引脚的自动识别、间距精准测量、缺陷判定与结果分类。

## 系统特点

- **传统视觉方案**：基于OpenCV传统图像处理，无需深度学习GPU，部署成本低
- **工业场景优化**：针对光照不均、引脚密集、微小尺寸、摆放偏移等实际工况针对性优化
- **模块化架构**：预处理->检测->测量->判定各环节独立可调
- **批量处理**：支持文件夹批量检测，自动生成报告
- **可视化界面**：基于Tkinter的图形界面的，实时显示检测过程和结果

## 项目结构

```
├── main.py                          # 系统入口
├── requirements.txt                 # 依赖清单
├── config/
│   └── default_config.yaml          # 默认配置文件
├── src/
│   ├── app_controller.py            # 应用主控协调器
│   ├── image_processing/
│   │   ├── preprocess.py            # 图像预处理
│   │   ├── pin_detection.py         # 引脚检测
│   │   └── roi_extraction.py        # ROI提取
│   ├── measurement/
│   │   ├── pitch_calculation.py     # 间距计算
│   │   └── calibration.py           # 相机标定
│   ├── classification/
│   │   └── defect_classifier.py     # 缺陷判定
│   └── gui/
│       └── main_window.py           # 主界面
├── docs/
│   └── architecture.md              # 架构文档
├── tests/                           # 测试目录
└── samples/                         # 样本图像目录
```

## 安装

```bash
pip install -r requirements.txt
```

## 使用

```bash
python main.py
```

### GUI操作

1. **文件 -> 打开图像** 或 Ctrl+O 加载连接器图像
2. **检测 -> 执行检测** 或 F5 运行检测流水线
3. 右侧面板显示检测信息和判定结果
4. **检测 -> 批量处理** 选择文件夹批量检测
5. **文件 -> 保存报告** 导出检测结果

## 配置

编辑 `config/default_config.yaml` 可调整：

- 图像预处理参数（去噪强度、CLAHE参数、二值化方法）
- 引脚检测参数（面积范围、长宽比、凸度阈值）
- 标定参数（像素当量）
- 判定参数（标称间距、公差）

## 标定方法

1. 拍摄已知实际尺寸的标定件
2. 测量图像中的像素长度
3. 在 config 中填入 `pixel_per_mm = 实际尺寸(mm) / 像素长度(px)`

## 依赖

- Python 3.8+
- OpenCV 4.9+
- NumPy
- PyYAML
- Pillow