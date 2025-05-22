import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# STEP 1: Define image list (hardcoded for loop)
# These are the filenames of your 10 input images
image_dir = "DIP Project"
image_files = [
    "cataract1.jpeg", "dry1.jpeg", "hyper1.jpeg", "mild1.jpeg", "moderate1.jpeg",
    "norm1.jpeg", "patho1.jpeg", "proliferate1.jpeg", "severe1.jpeg", "wet1.jpeg"
]

# STEP 2: Load both RGB and grayscale version of image
def load_grayscale_and_color(image_path):
    rgb_img = Image.open(image_path).convert('RGB')  # Load and ensure RGB format
    gray_img = rgb_img.convert('L')  # Convert to grayscale (luminance channel only)
    return np.array(rgb_img), np.array(gray_img)

# STEP 3: Piecewise Linear Contrast Stretching
def piecewise_linear_stretch(image, t, L_min=0, L_max=255):
    img = image.copy().astype(np.float32)   # float copy of image so we can modify original and do decimal math 
    I_min, I_max = np.min(img), np.max(img) # find min and max pixel intensities in range 
    result = np.zeros_like(img)             # Placeholder output image

    # Create masks for pixels <= t and > t
    mask1 = img <= t # pixels in low region 
    mask2 = img > t # high region 

    # Apply piecewise formula for pixels <= t
    if t != I_min:
        result[mask1] = ((img[mask1] - I_min) * (t - L_min) / (t - I_min)) + L_min
    else:
        result[mask1] = L_min

    # Apply piecewise formula for pixels > t
    if t != I_max:
        result[mask2] = ((img[mask2] - t + 1) * (L_max - t + 1) / (I_max - t + 1)) + t + 1
    else:
        result[mask2] = L_max

    return np.clip(result, 0, 255).astype(np.uint8)  # Ensure values valid 8 bit range [0-225] and converts back to integers

# STEP 4: Compute EME (Enhancement Measure Estimation)
# Measures contrast based on local patches
def compute_eme(image, block_size=(8, 8)):
    h, w = image.shape
    M, N = block_size
    eme_sum, count = 0, 0

    # Slide through the image in MxN blocks
    for i in range(0, h, M):
        for j in range(0, w, N):
            block = image[i:i+M, j:j+N]
            if block.size == 0:
                continue  # Skip empty blocks
            I_min, I_max = np.min(block), np.max(block)
            if I_min == 0:  # Avoid division by 0
                continue
            eme_sum += 20 * np.log10(I_max / I_min)  # Add local EME
            count += 1

    return eme_sum / count if count else 0  # Return average EME

# MAIN PROCESSING PIPELINE
if __name__ == "__main__":
    for filename in image_files:
        path = f"Original_Images/{filename}"  # path relative to subfolder
        rgb, gray = load_grayscale_and_color(path)  # load image

        emes = []  # Store EME values per threshold
        thresholds = list(range(0, 256))  # Thresholds from 0 to 255

        # Try contrast stretching for every threshold
        for t in thresholds:
            stretched = piecewise_linear_stretch(gray, t)
            eme = compute_eme(stretched)
            emes.append(eme)

        # Select best threshold that gives max EME
        best_t = thresholds[np.argmax(emes)]
        best_eme = max(emes)
        best_image = piecewise_linear_stretch(gray, best_t)

       
        # Display comparison images
        plt.figure(figsize=(10, 4))

        # Left: original RGB image
        plt.subplot(1, 2, 1)
        plt.imshow(rgb)
        plt.title(f'Original (Color): {filename}')
        plt.axis('off')

        # Right: enhanced grayscale result
        plt.subplot(1, 2, 2)
        plt.imshow(best_image, cmap='gray')
        plt.title(f'Enhanced (t={best_t})')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

       
        # Plot EME(t) vs threshold t
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

        # Print summary
        print(f">>> {filename}: Optimal Threshold = {best_t}, Max EME = {round(best_eme, 2)}\n")
