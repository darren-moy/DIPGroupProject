import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def apply_median_filter(image):
    rows, cols = image.shape
    filtered_image = image.copy()
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            window = image[i - 1:i + 2, j - 1:j + 2]
            sorted_window = np.sort(window.flatten())
            filtered_image[i, j] = sorted_window[5]
    return filtered_image

def linearContrastStretching(img):
    img = img.astype(np.float64)
    minVal, maxVal = img.min(), img.max()
    L_min, L_max = 0, 255
    stretched = ((img - minVal) / (maxVal - minVal)) * (L_max - L_min) + L_min
    return stretched.astype(np.uint8)

def convolve(image, kernel):
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    h, w = image.shape
    result = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            region = padded[i:i + k_h, j:j + k_w]
            result[i, j] = np.sum(region * kernel)
    
    return result

def edge_magnitude(gray, kx, ky):
    gx = convolve(gray, kx)
    gy = convolve(gray, ky)
    mag = np.sqrt(gx**2 + gy**2)
    mag = np.clip(mag / mag.max() * 255, 0, 255)
    return mag.astype(np.uint8)

def local_adaptive_thresholding(image, k = .9, window_size = 7):
    rows, cols = image.shape

    # Initialize binarized image
    binarized_image = np.zeros((rows, cols), dtype=np.uint8)
    M = 255
    for i in range(rows):
        for j in range(cols):
            # Define the local window around the pixel (ensure the window stays within bounds)
            x_min = max(i - window_size // 2, 0)
            x_max = min(i + window_size // 2 + 1, rows)
            y_min = max(j - window_size // 2, 0)
            y_max = min(j + window_size // 2 + 1, cols)

            # Extract local window
            local_window = image[x_min:x_max, y_min:y_max]

            # Calculate local
            local_mean = np.mean(local_window)
            I_max = np.max(local_window)
            I_min = np.min(local_window)

            # Calculate threshold
            T = k * (local_mean + (I_max - I_min) / M)

            # Apply threshold
            binarized_image[i, j] = 255 if image[i, j] > T else 0
    
    return binarized_image

def calculate_psnr(original, modified, max_pixel):
    mse = np.mean((original - modified) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr

# Sobel (3x3)
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]])

image_files = [
    "cataract1.jpeg", "dry1.jpeg", "hyper1.jpeg", "mild1.jpeg", "moderate1.jpeg",
    "norm1.jpeg", "patho1.jpeg", "proliferate1.jpeg", "severe1.jpeg", "wet1.jpeg"
]

image = cv2.imread('Original_Images\\cataract1.jpeg', cv2.IMREAD_GRAYSCALE)

preprocessed = apply_median_filter(image)
preprocessed = linearContrastStretching(preprocessed)
preprocessed = edge_magnitude(preprocessed, sobel_x, sobel_y)

binarized_image1 = local_adaptive_thresholding(preprocessed, .9)
pnsr1 = calculate_psnr(image, binarized_image1, 255)

binarized_image2 = local_adaptive_thresholding(preprocessed, .5)
pnsr2 = calculate_psnr(image, binarized_image2, 255)


binarized_image3 = local_adaptive_thresholding(preprocessed, .1)


# Different K
plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.imshow(image, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(binarized_image1, cmap='gray')
plt.title('k = 0.9')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(binarized_image2, cmap='gray')
plt.title('k = 0.5')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(binarized_image3, cmap='gray')
plt.title('k = 0.1')
plt.axis('off')

plt.tight_layout()
plt.show()

for filename in image_files:
    path = f"Original_Images/{filename}"
    if not os.path.exists(path):
        print(f"Image not found: {path}")
        continue

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    preprocessed = apply_median_filter(image)
    preprocessed = linearContrastStretching(preprocessed)
    preprocessed = edge_magnitude(preprocessed, sobel_x, sobel_y)

    k = .1
    binarized_image = local_adaptive_thresholding(preprocessed, k)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title(f'{filename} Original')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(preprocessed, cmap='gray')
    plt.title(f'{filename} Preprocessed')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(binarized_image, cmap='gray')
    plt.title(f'{filename} k = {k}')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    psnr = calculate_psnr(image, binarized_image, 255)
    print(f'{filename}: k = {k}, PSNR = {psnr : .2f}')

