import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# define image list
image_dir = "DIP Project"
image_files = [
    "cataract1.jpeg", "dry1.jpeg", "hyper1.jpeg", "mild1.jpeg", "moderate1.jpeg",
    "norm1.jpeg", "patho1.jpeg", "proliferate1.jpeg", "severe1.jpeg", "wet1.jpeg"
]

# loading images
def load_grayscale_and_color(image_path):
    rgb_img = Image.open(image_path).convert('RGB')
    gray_img = rgb_img.convert('L')
    return np.array(rgb_img), np.array(gray_img)

# piecewise contrast stretching
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
        result[mask2] = ((img[mask2] - t) * (L_max - t) / (I_max - t)) + t
    else:
        result[mask2] = L_max

    return np.clip(result, 0, 255).astype(np.uint8)

# computing EME
def compute_eme(image, block_size=(8, 8)):
    h, w = image.shape
    M, N = block_size
    eme_sum, count = 0, 0

    for i in range(0, h, M):
        for j in range(0, w, N):
            block = image[i:i+M, j:j+N]
            if block.size == 0:
                continue
            I_min, I_max = np.min(block), np.max(block)
            if I_min == 0:
                continue
            eme_sum += 20 * np.log10(I_max / I_min)
            count += 1

    return eme_sum / count if count else 0

# pipeline for each image
for filename in image_files:
    path = f"Original_Images/{filename}"
    rgb, gray = load_grayscale_and_color(path)

    emes = []
    thresholds = list(range(0, 256))

    for t in thresholds:
        stretched = piecewise_linear_stretch(gray, t)
        eme = compute_eme(stretched)
        emes.append(eme)

    best_t = thresholds[np.argmax(emes)]
    best_eme = max(emes)
    best_image = piecewise_linear_stretch(gray, best_t)

    # compare color and new image
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(rgb)  # original RGB image
    plt.title(f'Original (Color): {filename}')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(best_image, cmap='gray')
    plt.title(f'Enhanced (t={best_t})')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # plot EME curve
    plt.figure(figsize=(8, 4))
    plt.plot(thresholds, emes, label='EME(t)')
    plt.axvline(best_t, color='r', linestyle='--', label=f'Optimal t = {best_t}')
    plt.title(f'EME vs t — {filename}')
    plt.xlabel('Threshold t')
    plt.ylabel('EME')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f">>> {filename}: Optimal Threshold = {best_t}, Max EME = {round(best_eme, 2)}\n")