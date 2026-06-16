# CNN 活体检测训练优化计划

> 说明：当前 GitHub 连接器可直接读取根目录文本文件，但训练代码主要在 `活体检测训练脚本.zip` 中，未直接解压分析。因此本文件是基于当前 README、公开脚本、既有训练流程经验和论文项目需求整理的优化计划。

## 1. 代码结构优化

建议将训练脚本压缩包解压后整理为：

```text
Liveness-CNN-detection/
├── README.md
├── requirements-minimal.txt
├── requirements-train.txt
├── data/
│   ├── real/
│   └── fake/
├── scripts/
│   ├── camera_gather.py
│   ├── train.py
│   ├── evaluate.py
│   └── liveness_demo.py
├── liveness/
│   ├── model.py
│   ├── dataset.py
│   ├── metrics.py
│   └── utils.py
├── models/
└── outputs/
```

## 2. 数据集优化

### 2.1 不建议只按帧随机切分

如果从同一个视频中抽帧，再随机分成训练集和测试集，容易出现“同一段视频的相邻帧同时进入训练和测试”。这样会导致测试准确率虚高。

推荐：

- 按视频切分，而不是按帧切分。
- 同一个人的同一段视频只能属于 train / val / test 中的一个集合。
- 建议比例：70% train、15% val、15% test。

### 2.2 增加跨设备测试

至少补充：

- 手机前置摄像头。
- 笔记本摄像头。
- 外接 USB 摄像头。
- 弱光、强光、背光环境。
- 纸质照片、屏幕视频、二次录屏攻击。

## 3. 训练脚本优化

建议 `train.py` 增加：

1. `--seed` 固定随机种子。
2. `--val-split` 和 `--test-split`。
3. `--epochs`、`--batch-size`、`--lr` 命令行参数。
4. `EarlyStopping`。
5. `ModelCheckpoint` 保存最优模型。
6. `ReduceLROnPlateau` 自动降低学习率。
7. 混淆矩阵、Precision、Recall、F1 输出。
8. 单独 `evaluate.py` 对外部测试集评估。

推荐回调：

```python
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "models/best_liveness.keras",
        monitor="val_loss",
        save_best_only=True,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-7,
    ),
]
```

## 4. 模型结构优化

如果当前是浅层 CNN，可以继续保留轻量方向，但建议：

1. 使用 BatchNorm 稳定训练。
2. 使用 Dropout 抑制过拟合。
3. 使用 GlobalAveragePooling2D 替代大规模 Flatten。
4. 控制输入尺寸，例如 64x64 或 96x96。
5. 二分类输出建议使用 sigmoid + binary_crossentropy。
6. 如果使用 softmax 两分类，也可以，但标签和 loss 要保持一致。

推荐轻量结构：

```text
Input 96x96x3
Conv 3x3 16 + BN + ReLU
MaxPool
Conv 3x3 32 + BN + ReLU
MaxPool
Conv 3x3 64 + BN + ReLU
GlobalAveragePooling
Dropout 0.4
Dense 1 sigmoid
```

## 5. 推理优化

1. 导出 `.keras` 或 `.h5` 模型作为训练版本。
2. 导出 TFLite 作为移动端/边缘端版本。
3. 使用 int8 或 float16 量化。
4. 在 demo 中增加连续帧投票，避免单帧抖动。
5. 对人脸 ROI 做固定尺寸 resize，并统一归一化。

## 6. 论文写作建议

可以在论文中诚实表达：

- 当前项目以轻量、低算力部署为目标。
- 初始数据集上准确率很高，但存在数据来源单一导致的泛化风险。
- 后续通过独立测试集、跨设备测试、更多攻击样本来验证泛化能力。
- 使用量化、剪枝、轻量 CNN 结构以适配 CPU / 边缘设备。

不建议写：

- “模型已经可以完全防御所有攻击”。
- “准确率 100% 代表泛化能力完美”。
- “只用一段视频抽帧测试即可证明模型可靠”。
