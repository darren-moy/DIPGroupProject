import numpy as np
from PIL import Image
import os
import math

# Median Filter
def median_filter_manual(image, kernel_size=3):
    pad_size = kernel_size // 2
    padded = np.pad(image, pad_size, mode='edge')
    output = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+kernel_size, j:j+kernel_size]
            output[i, j] = np.median(region)
    return output.astype(np.uint8)

# Preprocessing Step 2: Contrast Stretching
def piecewise_linear_stretch(image, t, L_min=0, L_max=255):
    img = image.copy().astype(np.float32)
    I_min, I_max = np.min(img), np.max(img)
    result = np.zeros_like(img)
    mask1 = img <= t
    mask2 = img > t
    if t != I_min:
        result[mask1] = ((img[mask1] - I_min) * (t - L_min) / (t - I_min)) + L_min
    else:
        result[mask1] = L_min
    if t != I_max:
        result[mask2] = ((img[mask2] - t + 1) * (L_max - t + 1) / (I_max - t + 1)) + t + 1
    else:
        result[mask2] = L_max
    return np.clip(result, 0, 255).astype(np.uint8)

# Preprocessing Step 3: Sobel Edge Enhancement 
def sobel_edge_enhancement(image):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    padded = np.pad(image, 1, mode='edge')
    output = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+3, j:j+3]
            Gx = np.sum(Kx * region)
            Gy = np.sum(Ky * region)
            output[i, j] = np.sqrt(Gx**2 + Gy**2)
    return np.clip(output, 0, 255).astype(np.uint8)

# Otsu's Thresholding 
def otsu_threshold(image):
    hist, _ = np.histogram(image, bins=256, range=(0, 256))
    total = image.size
    probabilities = hist / total
    max_variance = 0
    threshold = 0
    between_variance = np.zeros(256)
    for t in range(256):
        w0 = np.sum(probabilities[:t+1])
        w1 = np.sum(probabilities[t+1:])
        if w0 == 0 or w1 == 0:
            continue
        mu0 = np.sum(np.arange(t+1) * probabilities[:t+1]) / w0
        mu1 = np.sum(np.arange(t+1, 256) * probabilities[t+1:]) / w1
        var_between = w0 * w1 * (mu0 - mu1) ** 2
        between_variance[t] = var_between
        if var_between > max_variance:
            max_variance = var_between
            threshold = t
    return threshold, between_variance

# PSNR Calculation
def compute_psnr(original, processed):
    mse = np.mean((original.astype(np.float32) - processed.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')  # perfect match
    max_pixel = 255.0
    psnr = 10 * math.log10((max_pixel ** 2) / mse)
    return psnr

# Main Execution 
image_files = [
    "cataract1.jpeg", "dry1.jpeg", "hyper1.jpeg", "mild1.jpeg", "moderate1.jpeg",
    "norm1.jpeg", "patho1.jpeg", "proliferate1.jpeg", "severe1.jpeg", "wet1.jpeg"
]

results = []

for filename in image_files:
    path = f"Original_Images/{filename}"
    if not os.path.exists(path):
        print(f"Image not found: {path}")
        continue

    # Load grayscale image
    rgb = Image.open(path).convert("RGB")
    gray = rgb.convert("L")
    gray_np = np.array(gray)

    filtered = median_filter_manual(gray_np)
    stretched = piecewise_linear_stretch(filtered, t=128)
    edge_enhanced = sobel_edge_enhancement(stretched)

    # Apply Otsu
    t_otsu, var_plot = otsu_threshold(edge_enhanced)
    segmented = edge_enhanced > t_otsu
    segmented = segmented.astype(np.uint8) * 255  # Convert to display format

    # Calculate PSNR 
    psnr_value = compute_psnr(gray_np, segmented)

    # Collect results
    results.append((filename, t_otsu, round(psnr_value, 2)))
    print(f"{filename}: Otsu t = {t_otsu}, PSNR = {round(psnr_value, 2)} dB")

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(gray_np, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(edge_enhanced, cmap='gray')
    plt.title('Preprocessed')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(segmented, cmap='gray')
    plt.title(f'Otsu Segmented (t={t_otsu})')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Optional: Print Summary Table
print("\nSummary Table:")
for name, t, psnr in results:
    print(f"{name:20} | Otsu Threshold: {t:3d} | PSNR: {psnr} dB")
