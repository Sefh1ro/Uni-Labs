#!/usr/bin/env python3
"""
lab5_gaussian.py
ЛR-5: grayscale -> add gaussian noise -> denoise with Gaussian filter (separable)
         -> Laplacian-of-Gaussian (LoG) for edges
Вимоги: Python 3, Pillow, numpy
Встановлення: pip install pillow numpy
Запуск: python lab5_gaussian.py input.jpg
"""
import sys, time
import numpy as np
from PIL import Image, ImageOps, ImageFilter


def to_grayscale(img):
    return ImageOps.grayscale(img)

def image_to_array(img):
    return np.asarray(img).astype(np.float32)

def array_to_image(a):
    a = np.clip(a, 0, 255).astype(np.uint8)
    return Image.fromarray(a)

def add_gaussian_noise(img_arr, sigma):
    noise = np.random.normal(0.0, sigma, img_arr.shape).astype(np.float32)
    return img_arr + noise

def gaussian_kernel_1d(sigma, radius=None):
    if radius is None:
        radius = max(1, int(3.0 * sigma))
    x = np.arange(-radius, radius+1)
    k = np.exp(-(x**2) / (2 * sigma * sigma))
    k = k / k.sum()
    return k

def separable_gaussian_filter(img_arr, sigma):
    """Apply separable gaussian filter to 2D array (grayscale)."""
    k = gaussian_kernel_1d(sigma)
    r = k.size // 2
    # convolve rows
    tmp = np.zeros_like(img_arr, dtype=np.float32)
    H, W = img_arr.shape
    # pad horizontally
    pad = r
    img_padded = np.pad(img_arr, ((0,0),(pad,pad)), mode='reflect')
    for i in range(W):
        tmp[:, i] = np.sum(img_padded[:, i:i+2*r+1] * k[np.newaxis, :], axis=1)
    # convolve cols
    out = np.zeros_like(tmp, dtype=np.float32)
    img_padded = np.pad(tmp, ((pad,pad),(0,0)), mode='reflect')
    for j in range(H):
        out[j, :] = np.sum(img_padded[j:j+2*r+1, :] * k[:, np.newaxis], axis=0)
    return out

def laplacian_of_gaussian_kernel(sigma, radius=None):
    if radius is None:
        radius = max(1, int(3.0 * sigma))
    x = np.arange(-radius, radius+1)
    y = x[:, None]
    rsq = x[None,:]**2 + y**2
    sigma2 = sigma*sigma
    factor = (rsq - 2*sigma2) / (sigma2**2)
    kernel = factor * np.exp(-rsq / (2*sigma2))
    kernel -= kernel.mean()  # zero-sum
    return kernel

def convolve2d(img_arr, kernel):
    kr, kc = kernel.shape
    rr = kr // 2; rc = kc // 2
    H, W = img_arr.shape
    out = np.zeros_like(img_arr, dtype=np.float32)
    padded = np.pad(img_arr, ((rr,rr),(rc,rc)), mode='reflect')
    for i in range(H):
        for j in range(W):
            patch = padded[i:i+kr, j:j+kc]
            out[i,j] = np.sum(patch * kernel)
    return out

def psnr(orig, other):
    orig = np.asarray(orig, dtype=np.float32)
    other = np.asarray(other, dtype=np.float32)
    mse = np.mean((orig - other) ** 2)
    if mse == 0:
        return float('inf')
    PIXMAX = 255.0
    return 20 * np.log10(PIXMAX / np.sqrt(mse))

def demo(in_path, out_prefix="lab5_out", noise_sigma=20.0, gauss_sigma=1.5, log_sigma=1.0):
    img = Image.open(in_path).convert("RGB")
    gray = to_grayscale(img)
    gray_arr = image_to_array(gray)  # shape (H,W)
    gray_arr_orig = gray_arr.copy()

    # 1) add gaussian noise
    np.random.seed(0)
    noisy = add_gaussian_noise(gray_arr, noise_sigma)
    noisy_img = array_to_image(noisy)
    noisy_img.save(f"{out_prefix}_noisy.png")

    # PSNR noisy vs orig
    p_noisy = psnr(gray_arr_orig, noisy)
    print(f"Noise sigma={noise_sigma}: PSNR(noisy, orig) = {p_noisy:.2f} dB")

    # 2) Gaussian denoising (separable)
    t0 = time.perf_counter()
    denoised = separable_gaussian_filter(noisy, gauss_sigma)
    t1 = time.perf_counter()
    denoised_img = array_to_image(denoised)
    denoised_img.save(f"{out_prefix}_denoised_gauss_sigma{gauss_sigma:.2f}.png")
    p_d = psnr(gray_arr_orig, denoised)
    print(f"Gaussian filter sigma={gauss_sigma}: time={ (t1-t0)*1000:.1f} ms, PSNR = {p_d:.2f} dB")

    # 3) Laplacian of Gaussian (for edges/differentiation)
    log_kernel = laplacian_of_gaussian_kernel(log_sigma)
    t0 = time.perf_counter()
    edges = convolve2d(denoised, log_kernel)  # usually apply LoG on smoothed image
    t1 = time.perf_counter()
    # Normalize edges for visualization
    edges_vis = (edges - edges.min()) / (edges.max() - edges.min() + 1e-12) * 255.0
    array_to_image(edges_vis).save(f"{out_prefix}_LoG_edges_sigma{log_sigma:.2f}.png")
    print(f"LoG sigma={log_sigma}: kernel_size={log_kernel.shape}, time={ (t1-t0)*1000:.1f} ms")

    # 4) Save original grayscale
    array_to_image(gray_arr_orig).save(f"{out_prefix}_orig_gray.png")

    print("Saved files:", f"{out_prefix}_orig_gray.png", f"{out_prefix}_noisy.png",
          f"{out_prefix}_denoised_gauss_sigma{gauss_sigma:.2f}.png",
          f"{out_prefix}_LoG_edges_sigma{log_sigma:.2f}.png")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python lab5_gaussian.py input_image.jpg")
        sys.exit(1)
    in_path = sys.argv[1]
    demo(in_path)
