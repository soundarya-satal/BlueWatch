import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------
# Utility functions
# -----------------------------
def compute_contrast_metric(gray):
    # Standard deviation is a simple contrast measure
    return float(np.std(gray))

def compute_noise_metric(gray):
    # Noise estimate using Laplacian variance (higher can mean more detail OR noise)
    lap = cv.Laplacian(gray, cv.CV_64F)
    return float(lap.var())

def gamma_correction(img, gamma):
    inv = 1.0 / gamma
    lut = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype("uint8")
    return cv.LUT(img, lut)

def unsharp_mask(img, sigma=1.2, amount=1.3, threshold=0):
    blur = cv.GaussianBlur(img, (0, 0), sigma)
    sharp = cv.addWeighted(img, 1 + amount, blur, -amount, 0)

    if threshold > 0:
        # Only sharpen where difference is above threshold (reduces noise sharpening)
        low_contrast_mask = np.abs(img.astype(np.int16) - blur.astype(np.int16)) < threshold
        sharp[low_contrast_mask] = img[low_contrast_mask]
    return sharp

def plot_hist(ax, img, title):
    hist = cv.calcHist([img], [0], None, [256], [0, 256]).ravel()
    ax.plot(hist)
    ax.set_title(title)
    ax.set_xlim([0, 255])

# -----------------------------
# Advanced enhancement pipeline
# -----------------------------
def enhance_image_advanced(img, use_nlmeans=True):
    # If color, convert to LAB and enhance L channel (best practice)
    is_color = (len(img.shape) == 3 and img.shape[2] == 3)

    if is_color:
        lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
        L, A, B = cv.split(lab)
        gray = L
    else:
        gray = img.copy()

    contrast = compute_contrast_metric(gray)
    noise = compute_noise_metric(gray)

    # 1) Edge-preserving denoise
    # Bilateral is great to reduce noise while keeping edges
    # Strength based on noise/contrast
    d = 9
    sigmaColor = 50 if noise < 200 else 75
    sigmaSpace = 50
    den_bilat = cv.bilateralFilter(gray, d, sigmaColor, sigmaSpace)

    # Optional: NLMeans (stronger but slower)
    if use_nlmeans:
        # h based on contrast (low contrast => slightly higher denoise)
        h = 7 if contrast > 40 else 10
        den = cv.fastNlMeansDenoising(den_bilat, None, h=h, templateWindowSize=7, searchWindowSize=21)
    else:
        den = den_bilat

    # 2) Adaptive CLAHE
    # Lower contrast => higher clipLimit
    clip = 1.5 if contrast > 50 else 2.5
    tile = (8, 8) if gray.shape[0] > 300 else (6, 6)
    clahe = cv.createCLAHE(clipLimit=clip, tileGridSize=tile)
    con = clahe.apply(den)

    # 3) Adaptive sharpening (with threshold to avoid noise boost)
    # Low contrast => stronger sharpening
    amount = 0.8 if contrast > 60 else 1.4
    sharp = unsharp_mask(con, sigma=1.2, amount=amount, threshold=3)

    # 4) Gamma based on brightness
    mean_intensity = float(np.mean(sharp))
    if mean_intensity < 90:
        gamma = 0.75   # brighten
    elif mean_intensity > 160:
        gamma = 1.2    # darken slightly
    else:
        gamma = 1.0
    out = gamma_correction(sharp, gamma) if gamma != 1.0 else sharp

    # If original was color, put enhanced L back and convert to BGR
    if is_color:
        lab_enh = cv.merge([out, A, B])
        final = cv.cvtColor(lab_enh, cv.COLOR_LAB2BGR)
        return gray, den, con, sharp, out, final, (contrast, noise, clip, tile, gamma)
    else:
        return gray, den, con, sharp, out, out, (contrast, noise, clip, tile, gamma)

# -----------------------------
# Runner
# -----------------------------
def run():
    root = os.getcwd()
    imgPath = os.path.join(root, "badQuality.jpeg")

    img = cv.imread(imgPath, cv.IMREAD_UNCHANGED)
    if img is None:
        print("Image not found:", imgPath)
        return

    # If image has alpha, drop alpha
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    gray, den, con, sharp, out, final, stats = enhance_image_advanced(img, use_nlmeans=True)
    contrast, noise, clip, tile, gamma = stats

    # Display results
    plt.figure(figsize=(16, 8))

    # Images row
    titles = [
        "Original (Gray/L channel)",
        "Denoised (edge-preserving)",
        f"CLAHE adaptive (clip={clip}, tile={tile})",
        "Sharpened (unsharp + threshold)",
        f"Final (gamma={gamma:.2f})"
    ]
    imgs = [gray, den, con, sharp, out]

    for i in range(5):
        ax = plt.subplot(2, 5, i + 1)
        ax.imshow(imgs[i], cmap="gray")
        ax.set_title(titles[i])
        ax.axis("off")

    # Histograms row
    for i in range(5):
        ax = plt.subplot(2, 5, 5 + i + 1)
        plot_hist(ax, imgs[i], f"Histogram: {i+1}")

    plt.suptitle(
        f"Stats → contrast(std)={contrast:.2f}, noise(lap var)={noise:.2f}",
        fontsize=12
    )
    plt.tight_layout()
    plt.show()

    # If color, show original vs final color
    if len(img.shape) == 3:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        plt.title("Original (Color)")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(cv.cvtColor(final, cv.COLOR_BGR2RGB))
        plt.title("Final Enhanced (Color)")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    run()
