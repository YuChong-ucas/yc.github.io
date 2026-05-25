---
title: 基于pp-ocr,从 SVTR logits 中估算字符 bbox
author: yuchong
date: 2026-04-27 00:34:00 +0800
categories: [AI]
tags: [AI]
math: true
---

一般调用pp-ocr模型，我们会得到按照行输出的bbox,但是如果希望按照单字输出，就需要从SVTR logits 中估算字符 bbox。

【应用场景】

  1.为了单字检测，在AIGC项目中并进行改字；

  2.解决生图模型后处理中的小字崩坏问题，结合
  https://github.com/limuloo/RefineAnything
  这个开源项目，就可以实现对小字崩坏的修复；

  3.训练模型时，为了缓解小字崩坏，在输入的时候，就可以在提示词里加入小字，此时可以进行单字检测后进行加入，并带有精确的bbox。

```python

input_tensor, img_width_rec = preprocess_for_recognizer(crop_img)
print(f"Input shape: {input_tensor.shape}") 
input_paddle = paddle.to_tensor(input_tensor, place=paddle.CPUPlace())
outputs = recognizer.predictor.run([input_paddle])
logits = outputs[0].numpy()
rec_result = recognizer.postprocess_op(logits)
if not rec_result:
      continue
text, score = rec_result[0]
T = logits.shape[1]
print(f"logits shape: {logits.shape}, inferred W = {T * 4}, actual W = {img_width_rec}")
char_boxes_rec = get_char_boxes_from_logits(logits, text, img_width_rec, 48)

```

其中的：
```python

def preprocess_for_recognizer(img):
    """
    复现 PaddleOCR 的 recognizer 预处理逻辑
    输入: HWC uint8 BGR image (e.g., from get_rotate_crop_image)
    输出: CHW float32 tensor ready for model input, and the resized width
    """
    # 确保是 3 通道
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    h, w = img.shape[:2]
    # 固定高度为 48，宽度按比例缩放（最小 32）
    new_h = 72#48
    # new_w = max(32, int(w * new_h / h))
    new_w = max(48, int(w * new_h / h))
    
    resized = cv2.resize(img, (new_w, new_h))
    # 归一化: (x / 255 - 0.5) / 0.5
    normalized = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
    # 转 CHW + batch
    chw = normalized.transpose(2, 0, 1)
    input_tensor = chw[np.newaxis, :]  # [1, 3, 48, new_w]
    
    return input_tensor, new_w



def get_char_boxes_from_logits(logits, text, img_width, img_height=48):
    """
    从 SVTR logits 中估算字符 bbox（无需 attention，仅用解码路径）

    Args:
        logits: [1, T, num_classes] - 模型原始输出
        text: str - 识别出的文本
        img_width: int - 裁剪图像宽度（如 320）
        img_height: int - 裁剪图像高度（固定为 48）

    Returns:
        List[[x1, y1, x2, y2]] - 字符 bbox 列表
    """
    T = logits.shape[1]  # 时间步数，通常 T = img_width // 4 (e.g., 320//4=80)

    # Step 1: 执行 greedy decode 并记录每个字符对应的时间步
    # probs = np.softmax(logits[0], axis=-1)
    x = logits[0]
    x_stable = x - np.max(x, axis=-1, keepdims=True)
    probs = np.exp(x_stable) / np.sum(np.exp(x_stable), axis=-1, keepdims=True)

    pred_indices = np.argmax(probs, axis=-1)  # [T]

    # 移除 blank (index=0) 和重复
    char_indices = []
    prev_idx = 0
    for t, idx in enumerate(pred_indices):
        if idx != 0 and idx != prev_idx:
            char_indices.append((idx, t))
        prev_idx = idx
   
    # 确保字符数匹配（防止解码误差）
    if len(char_indices) != len(text):
       
        # fallback: 等分
        char_width = img_width / len(text)
        return [[int(i * char_width), 0, int((i + 1) * char_width), img_height]
                for i in range(len(text))]

    # Step 2: 将时间步 t 映射到像素 x 坐标
    char_boxes = []
    for i, (char_idx, t) in enumerate(char_indices):
        # 特征图位置 -> 像素位置（线性映射）
        x_center = (t / T) * img_width

        # 估算字符宽度：相邻字符中点距离
        if i == 0:
            next_t = char_indices[1][1] if len(char_indices) > 1 else T
            char_w = ((next_t - t) / T) * img_width * 0.995
        elif i == len(char_indices) - 1:
            prev_t = char_indices[i - 1][1]
            char_w = ((t - prev_t) / T) * img_width * 0.995
        else:
            prev_t = char_indices[i - 1][1]
            next_t = char_indices[i + 1][1]
            char_w = (((next_t - prev_t) / 2) / T) * img_width * 0.995

        x1 = max(0, int(x_center - char_w / 2))
        x2 = min(img_width, int(x_center + char_w / 2))
        y1, y2 = 0, img_height

        char_boxes.append([x1, y1, x2, y2])

    return char_boxes

```

这样就可以估算出单个字的box了。