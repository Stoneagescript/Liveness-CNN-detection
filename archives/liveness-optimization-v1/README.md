# 活体检测项目优化归档｜v1

> 归档日期：2026-06-16  
> 归档性质：不破坏原项目的优化方案存档  
> 原项目：Stoneagescript/Liveness-CNN-detection

## 1. 当前项目可见状态

仓库根目录当前可见内容包括：

- `README.md`：项目说明，强调这是一个用于人脸检测/识别前置条件的开源活体检测模型。
- `reflection.py`：基于皮肤纹理和反光特征的轻量活体检测脚本。
- `requirements.txt`：当前环境依赖锁定文件。
- `活体检测训练脚本.zip`：训练脚本压缩包，GitHub 连接器当前未直接解压分析。

## 2. 主要问题判断

### 2.1 reflection.py 的问题

当前 `reflection.py` 使用：

- OpenCV Haar Cascade 检测人脸。
- 固定阈值分析高亮反光区域。
- GLCM 灰度共生矩阵提取纹理特征。
- 使用手写阈值判断 Alive / Dead。

这个方案优点是轻量、容易运行，但存在明显局限：

1. Haar 人脸框在侧脸、弱光、遮挡、摄像头差异下不稳定。
2. `cv2.threshold(blurred, 70, 255, ...)` 是固定阈值，不适合不同光照。
3. `contourArea > 500` 对分辨率敏感，不同摄像头会误判。
4. GLCM 直接使用 256 灰阶，计算量偏高，对实时性不友好。
5. `print` 写在实时循环里，会拖慢运行，也不适合部署。
6. 判断逻辑只有单帧，没有时间平滑，容易闪烁。
7. 没有命令行参数，不方便换摄像头、调阈值、保存日志。
8. 没有 `if __name__ == "__main__"`，作为模块导入时会直接打开摄像头。

### 2.2 requirements.txt 的问题

当前依赖文件存在本地绝对路径 wheel：

- `dlib @ file:///F:/...`
- `insightface @ file:///F:/...`

这会导致其他电脑无法直接安装。同时依赖里包含很多非项目必要包，例如 Jupyter、nbconvert、torch、onnxruntime-gpu 等。如果项目核心是 OpenCV + TensorFlow / scikit-image，建议拆分依赖：

- `requirements-minimal.txt`：最小可运行依赖。
- `requirements-dev.txt`：开发/训练依赖。
- `requirements-gpu.txt`：可选 GPU 依赖。

## 3. 本次归档提供的优化内容

本目录新增：

- `reflection_optimized.py`：对原 `reflection.py` 的工程化优化版本。
- `requirements-minimal.txt`：轻量检测脚本最小依赖。
- `OPTIMIZATION_NOTES.md`：详细优化说明。
- `TRAINING_OPTIMIZATION_PLAN.md`：训练脚本压缩包解压后建议执行的模型训练优化路线。

## 4. 推荐下一步

1. 把 `活体检测训练脚本.zip` 解压到仓库标准目录，例如 `training/`。
2. 将训练、采集、推理脚本拆成：
   - `scripts/camera_gather.py`
   - `scripts/train.py`
   - `scripts/evaluate.py`
   - `scripts/liveness_demo.py`
3. 增加独立测试集，不要只看训练/验证曲线。
4. 增加混淆矩阵、Precision、Recall、F1。
5. 增加模型导出：`.h5`、`.tflite`、可选 ONNX。
6. 增加 README 的完整命令和数据目录结构。
