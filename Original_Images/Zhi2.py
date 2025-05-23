import cv2
import numpy as np
import matplotlib.pyplot as plt

image_files = [
    "cataract1.jpeg", "dry1.jpeg", "hyper1.jpeg", "mild1.jpeg", "moderate1.jpeg",
    "norm1.jpeg", "patho1.jpeg", "proliferate1.jpeg", "severe1.jpeg", "wet1.jpeg"
]

def adaptive_threshold(image_path, k=0.8, window_size=15):
    # 1. Load image and preprocess
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))  # Standardize size

    # 2. Preprocessing steps
    denoised = cv2.fastNlMeansDenoising(img, h=10)  # Noise reduction
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)  # Contrast enhancement
    edges = cv2.Canny(enhanced, 50, 150)  # Edge detection

    # 3. Adaptive threshold calculation
    padded = cv2.copyMakeBorder(enhanced, window_size // 2, window_size // 2,
                                window_size // 2, window_size // 2, cv2.BORDER_REFLECT)
    binary = np.zeros_like(enhanced)
    M = 255  # Max intensity

    for i in range(enhanced.shape[0]):
        for j in range(enhanced.shape[1]):
            window = padded[i:i + window_size, j:j + window_size]
            µ = np.mean(window)
            Imax = np.max(window)
            Imin = np.min(window)
            T = k * (µ + (Imax - Imin) / M)
            binary[i, j] = 255 if enhanced[i, j] > T else 0

    # 4. Postprocessing
    binary = cv2.bitwise_and(binary, 255 - edges)  # Combine with edges
    binary = cv2.medianBlur(binary, 3)  # Remove small noise

    return img, enhanced, binary


# Test on sample images
images = ["cataract1.jpeg", "dry1.jpeg", "hyper1.jpeg", "mild1.jpeg", "moderate1.jpeg",
    "norm1.jpeg", "patho1.jpeg", "proliferate1.jpeg", "severe1.jpeg", "wet1.jpeg"]
k_values = [0.7, 0.8, 0.9]  # Test different k values

for img_path in images:
    orig, enhanced, binary = adaptive_threshold(img_path)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1), plt.imshow(orig, cmap='gray'), plt.title("Original")
    plt.subplot(1, 3, 2), plt.imshow(enhanced, cmap='gray'), plt.title("Enhanced")
    plt.subplot(1, 3, 3), plt.imshow(binary, cmap='gray'), plt.title("Thresholded")
    plt.show()