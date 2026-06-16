# reflection.py 优化说明

## 1. 原脚本保留的核心思路

原脚本的核心思路是：

1. 用摄像头读取图像。
2. 用 Haar Cascade 找人脸。
3. 对人脸 ROI 做反光检测。
4. 对人脸 ROI 做 GLCM 纹理检测。
5. 反光和纹理都满足时判断为 Alive。

这个思路适合轻量级演示，也适合在没有深度学习模型、没有 GPU 的环境里做兜底检测。

## 2. 已优化点

### 2.1 固定阈值改成自适应阈值

原始版本使用：

```python
cv2.threshold(blurred, 70, 255, cv2.THRESH_BINARY)
```

问题是不同摄像头、不同光线下，70 这个阈值不稳定。

优化版改为：

```python
cv2.adaptiveThreshold(...)
```

这样对办公室、弱光、屏幕光照差异更稳。

### 2.2 绝对轮廓面积改成相对面积

原始版本使用：

```python
cv2.contourArea(cnt) > 500
```

问题是 500 像素在 640x480 和 1080p 摄像头下含义不同。

优化版改为：

```python
max_area / roi_area > reflect_area_ratio
```

这样更适合不同分辨率。

### 2.3 GLCM 灰阶从 256 降到 32

原始版本直接使用 256 灰阶 GLCM，计算量偏高。

优化版将 ROI 缩放到 96x96，并量化到 32 灰阶：

```python
glcm_levels = 32
```

这可以降低计算量，更适合实时摄像头检测。

### 2.4 增加时间平滑

单帧判断容易闪烁，优化版使用最近 N 帧投票：

```python
history = deque(maxlen=7)
stable_alive = alive_ratio >= 0.6
```

这样画面标签不会频繁 Alive / Spoof 来回跳。

### 2.5 增加命令行参数

优化版支持：

```bash
python reflection_optimized.py --camera 0 --debug
python reflection_optimized.py --reflect-area-ratio 0.01 --glcm-levels 32
```

方便不同设备调参。

### 2.6 增加 main 入口保护

优化版增加：

```python
if __name__ == "__main__":
    main()
```

这样脚本可以被其他代码导入，不会一导入就打开摄像头。

## 3. 仍然存在的局限

优化版仍然不是强安全级别活体检测，只适合轻量演示或低风险前置筛查。

它仍然可能被以下方式绕过：

- 高清屏幕播放真人视频。
- 强光环境下照片产生类似反光。
- 摄像头质量差导致纹理特征混乱。
- 只靠单 RGB 摄像头，缺少深度、红外、动作挑战。

## 4. 真正适合论文/产品继续优化的方向

1. 用 CNN 模型作为主判断，`reflection_optimized.py` 作为辅助特征。
2. 增加眨眼、点头、张嘴等交互式动作挑战。
3. 增加多帧时间模型，例如 3D CNN、LSTM、Temporal CNN。
4. 增加屏幕摩尔纹、边框、反射、频闪检测。
5. 使用更现代的人脸检测器替代 Haar，例如 MediaPipe Face Detection、YuNet 或 RetinaFace。
6. 增加独立测试集和跨设备测试集，避免只在自采数据上 100% 准确。
