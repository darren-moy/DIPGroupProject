import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure

from darren import image_files, piecewise_linear_stretch, load_grayscale_and_color, compute_eme


def rgb_to_grayscale(rgb_img):
    """Convert RGB to grayscale using standard coefficients"""
    return 0.299 * rgb_img[:, :, 0] + 0.587 * rgb_img[:, :, 1] + 0.114 * rgb_img[:, :, 2]


def normalize_channels(rgb_img):
    """Normalize RGB channels"""
    sum_channels = rgb_img.sum(axis=2)
    sum_channels[sum_channels == 0] = 1  # avoid division by zero
    r_norm = rgb_img[:, :, 0] / sum_channels
    g_norm = rgb_img[:, :, 1] / sum_channels
    b_norm = rgb_img[:, :, 2] / sum_channels
    return np.stack([r_norm, g_norm, b_norm], axis=2)


def gamma_correction(image, gamma=.7):
    """Apply gamma correction to image"""
    return exposure.adjust_gamma(image, gamma)


def alpha_blend(img1, img2, alpha=0.45):
    """Alpha blend two images"""
    return alpha * img1 + (1 - alpha) * img2


def student2_pipeline(rgb_img, optimal_t):
    """Full contrast enhancement pipeline for Student 2"""
    # Step 1: Convert to grayscale using standard method
    gray_img = rgb_to_grayscale(rgb_img)

    # Step 2: Normalize RGB channels
    normalized_rgb = normalize_channels(rgb_img)

    # Step 3: Create difference image and apply gamma correction
    diff_img = np.abs(normalized_rgb[:, :, 0] - normalized_rgb[:, :, 1])
    gamma_img = gamma_correction(diff_img, gamma=0.5)

    # Step 4: Alpha blend the grayscale and gamma corrected image
    blended_img = alpha_blend(gray_img / 255, gamma_img, alpha=0.5)
    blended_img = (blended_img * 255).astype(np.uint8)

    # Step 5: Apply optimal piecewise contrast enhancement
    final_img = piecewise_linear_stretch(blended_img, optimal_t)

    return final_img

if __name__ == "__main__":
    # Precomputed optimal thresholds and EMEs (from Student 1)
    optimal_thresholds = {
        "cataract1.jpeg": (127, 2.76),
        "dry1.jpeg": (122, 5.55),
        "hyper1.jpeg": (147, 6.71),
        "mild1.jpeg": (128, 6.14),
        "moderate1.jpeg": (128, 4.55),
        "norm1.jpeg": (127, 11.02),
        "patho1.jpeg": (128, 5.25),
        "proliferate1.jpeg": (121, 5.47),
        "severe1.jpeg": (122, 4.68),
        "wet1.jpeg": (110, 3.59)
    }
    for filename in image_files:
        path = f"Original_Images/{filename}"
        rgb, gray = load_grayscale_and_color(path)
    
        # Get precomputed optimal threshold and EME
        best_t, best_eme = optimal_thresholds[filename]
    
        # Student 2's processing
        enhanced_img = student2_pipeline(rgb, best_t)
    
        # Visualization
        plt.figure(figsize=(15, 5))
    
        plt.subplot(1, 3, 1)
        plt.imshow(rgb)
        plt.title(f'Original: {filename}')
        plt.axis('off')
    
        plt.subplot(1, 3, 2)
        plt.imshow(piecewise_linear_stretch(gray, best_t), cmap='gray')
        plt.title(f'Student 1 (t={best_t})')
        plt.axis('off')
    
        plt.subplot(1, 3, 3)
        plt.imshow(enhanced_img, cmap='gray')
        plt.title('Student 2 Enhanced')
        plt.axis('off')
    
        plt.tight_layout()
        plt.show()
    
        print(f">>> {filename}: Optimal Threshold = {best_t}, Max EME = {round(best_eme, 2)}\n")
