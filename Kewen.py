import numpy as np

from darren import image_files, load_grayscale_and_color
from Zhi import student2_pipeline

# Enhancement Measure Estimation (EME)
def compute_EME(img, k1=4, k2=16):
    rows, cols = img.shape
    blockRows = rows // k1
    blockCols = cols // k2
    eme = 0
    
    for k in range(k1):
        for l in range(k2):
            rowStart = k * blockRows
            rowEnd = (k + 1) * blockRows
            colStart = l * blockCols
            colEnd = (l + 1) * blockCols
            
            block = img[rowStart:rowEnd, colStart:colEnd]
            
            # Get I_Min and I_Max
            minVal = np.min(block)
            maxVal = np.max(block)
            
            # Prevent overflow
            maxVal = np.clip(maxVal, 0, 255)
            minVal = np.clip(minVal, 0, 255)
            
            # Avoid Division by 0
            if minVal == 0:
                minVal = 1
            eme += 20 * np.log(maxVal / minVal)
    
    return eme / (k1 * k2)


# Entropy-based EME (EMEE)
def compute_EMEE(img, k1=8, k2=8, alpha=1):
    rows, cols = img.shape
    blockRows = rows // k1
    blockCols = cols // k2
    emee = 0

    for k in range(k1):
        for l in range(k2):
            rowStart = k * blockRows
            rowEnd = (k + 1) * blockRows
            colStart = l * blockCols
            colEnd = (l + 1) * blockCols

            block = img[rowStart:rowEnd, colStart:colEnd]
            
            # Get I_Min and I_Max
            minVal = np.min(block)
            maxVal = np.max(block)
            
            # Prevent overflow
            maxVal = np.clip(maxVal, 0, 255)
            minVal = np.clip(minVal, 0, 255)
            
            # Avoid Division by 0
            if minVal == 0:
                continue
            emee += alpha * np.log(maxVal / minVal)
    
    return emee / (k1 * k2)

# Visibility (Michelson Contrast)
def compute_Visibility(img, k1=8, k2=8):
    rows, cols = img.shape
    blockRows = rows // k1
    blockCols = cols // k2
    visibility = 0

    for k in range(k1):
        for l in range(k2):
            rowStart = k * blockRows
            rowEnd = (k + 1) * blockRows
            colStart = l * blockCols
            colEnd = (l + 1) * blockCols

            block = img[rowStart:rowEnd, colStart:colEnd]
            
            # Get I_Min and I_Max
            minVal = np.min(block)
            maxVal = np.max(block)
            
            # Prevent overflow
            maxVal = np.clip(maxVal, 0, 255)
            minVal = np.clip(minVal, 0, 255)
            
            total = maxVal + minVal
            
            # Avoid Division by 0
            if total == 0:
                continue
            
            ratio = (maxVal - minVal) / total
            visibility += ratio
    
    return visibility

# Average Michelson EME (AME)
def compute_AME(img, k1=8, k2=8):
    rows, cols = img.shape
    blockRows = rows // k1
    blockCols = cols // k2
    ame = 0

    for k in range(k1):
        for l in range(k2):
            rowStart = k * blockRows
            rowEnd = (k + 1) * blockRows
            colStart = l * blockCols
            colEnd = (l + 1) * blockCols

            block = img[rowStart:rowEnd, colStart:colEnd]
            
            # Get I_Min and I_Max
            minVal = np.min(block)
            maxVal = np.max(block)
            
            # Prevent overflow
            maxVal = np.clip(maxVal, 0, 255)
            minVal = np.clip(minVal, 0, 255)
            
            total = maxVal + minVal
            
            # Avoid Division by 0
            if total == 0:
                continue
            
            ratio = (maxVal - minVal) / total
            ame += np.log(ratio)
    
    return -ame / (k1 * k2)

# Entropy-based AME (AMEE)
def compute_AMEE(img, k1=8, k2=8, alpha=1):
    rows, cols = img.shape
    blockRows = rows // k1
    blockCols = cols // k2
    amee = 0

    for k in range(k1):
        for l in range(k2):
            rowStart = k * blockRows
            rowEnd = (k + 1) * blockRows
            colStart = l * blockCols
            colEnd = (l + 1) * blockCols

            block = img[rowStart:rowEnd, colStart:colEnd]
            
            # Get I_Min and I_Max
            minVal = np.min(block)
            maxVal = np.max(block)
            
            # Prevent overflow
            maxVal = np.clip(maxVal, 0, 255)
            minVal = np.clip(minVal, 0, 255)
            
            total = maxVal + minVal
            
            # Avoid Division by 0
            if total == 0:
                continue
            
            ratio = (maxVal - minVal) / total
            amee += np.log(ratio**alpha)
    
    return -amee / (k1 * k2)

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
    # pipeline for each image
    for filename in image_files:
        path = f"Original_Images/{filename}"
        rgb, gray = load_grayscale_and_color(path)
    
        # Get precomputed optimal threshold and EME (from Student 1)
        best_t, best_eme = optimal_thresholds[filename]
    
        # Student 2's processing
        enhanced_img = student2_pipeline(rgb, best_t)
        
        # Compute metrics (Student 3)
        original_metrics = {
            "EME": compute_EME(gray),
            "EMEE": compute_EMEE(gray),
            "Visibility": compute_Visibility(gray),
            "AME": compute_AME(gray),
            "AMEE": compute_AMEE(gray)
        }
        
        enhanced_metrics = {
            "EME": compute_EME(enhanced_img),
            "EMEE": compute_EMEE(enhanced_img),
            "Visibility": compute_Visibility(enhanced_img),
            "AME": compute_AME(enhanced_img),
            "AMEE": compute_AMEE(enhanced_img)
        }
    
        # Print metrics in a table
        print(f"\n--- {filename} ---\n")
        print(f"{'Metric':<15} {'Original':<15} {'Enhanced':<15}")
        print("-" * 45) 
        
        for k in original_metrics:
            original_value = original_metrics[k]
            enhanced_value = enhanced_metrics[k]
            print(f"{k:<15} {original_value: <15.4f} {enhanced_value: <15.4f}")
    