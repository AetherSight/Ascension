import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_clothing_features(image_path):
    # 1. 加载图像 (RGB)
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # --- 方案 A: 灰度化 + CLAHE (推荐：保留花纹纹理) ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(gray)
    
    # --- 方案 B: Canny 边缘检测 (只看轮廓结构) ---
    # 先模糊降噪，再提边缘
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    img_canny = cv2.Canny(blurred, 50, 150)
    
    # --- 方案 C: Sobel 梯度图 (强调颜色变化的地方，即花纹) ---
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    img_grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    # --- 方案 D: 极端色彩抖动模拟 (人工染色训练) ---
    # 这种方案不需要改变图像格式，只需在训练时加入随机 Hue 变换
    # 这里模拟一个随机变色效果
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,0] = (hsv[:,:,0] + 40) % 180 # 偏移色相
    img_jitter = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # 显示对比图
    titles = ['Original', 'CLAHE (Texture)', 'Canny (Structure)', 'Sobel (Pattern)', 'Color Jitter']
    images = [img_rgb, img_clahe, img_canny, img_grad, img_jitter]
    
    plt.figure(figsize=(20, 10))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(images[i], cmap='gray' if len(images[i].shape)==2 else None)
        plt.title(titles[i])
        plt.axis('off')
    plt.show()

# 使用你的图片路径运行
process_clothing_features(r'S:\FFXIV_train_test\5.jpg')