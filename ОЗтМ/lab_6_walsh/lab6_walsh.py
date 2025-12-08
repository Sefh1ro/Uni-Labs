import numpy as np
from PIL import Image

# -------------------------------------------------------
# Hadamard matrix (recursive)
# -------------------------------------------------------
def hadamard(n: int) -> np.ndarray:
    """
    Build Hadamard matrix of order n (n must be power of 2)
    Elements are +1 / -1
    """
    if n == 1:
        return np.array([[1.0]])
    H = hadamard(n // 2)
    return np.block([
        [H,  H],
        [H, -H]
    ])

# -------------------------------------------------------
# Walsh–Hadamard 2D transform (orthonormal)
# -------------------------------------------------------
def walsh_2d(B: np.ndarray):
    """
    Forward 2D Walsh–Hadamard transform
    """
    N = B.shape[0]
    H = hadamard(N)
    Hn = H / np.sqrt(N)       # normalization
    U = Hn @ B @ Hn
    return U, Hn

def iwals_h_2d(U: np.ndarray, Hn: np.ndarray):
    """
    Inverse 2D Walsh–Hadamard transform
    """
    return Hn @ U @ Hn

# -------------------------------------------------------
# PSNR computation
# -------------------------------------------------------
def psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))

# -------------------------------------------------------
# Create test image 8x8 (letter "T")
# -------------------------------------------------------
B = np.array([
    [255,255,255,255,255,255,255,255],
    [0,0,0,255,255,0,0,0],
    [0,0,0,255,255,0,0,0],
    [0,0,0,255,255,0,0,0],
    [0,0,0,255,255,0,0,0],
    [0,0,0,255,255,0,0,0],
    [0,0,0,255,255,0,0,0],
    [0,0,0,255,255,0,0,0],
], dtype=np.float64)

# -------------------------------------------------------
# Walsh Transform
# -------------------------------------------------------
U, Hn = walsh_2d(B)
B_recon = iwals_h_2d(U, Hn)

# -------------------------------------------------------
# Diagnostics
# -------------------------------------------------------
energy = np.sum(B ** 2)
psnr_value = psnr(B, B_recon)

print("Total energy (sum of squares):", energy)
print("\nWalsh spectrum U (rounded):")
print(np.round(U, 2))
print("\nPSNR (recon vs original):", psnr_value)

# -------------------------------------------------------
# Save images for report
# -------------------------------------------------------
orig_img = np.clip(np.round(B), 0, 255).astype(np.uint8)
recon_img = np.clip(np.round(B_recon), 0, 255).astype(np.uint8)

# Visualize spectrum
U_abs = np.abs(U)
U_vis = (U_abs - U_abs.min()) / (np.ptp(U_abs) + 1e-12) * 255
U_vis = U_vis.astype(np.uint8)

Image.fromarray(orig_img).save("lab6_orig.png")
Image.fromarray(recon_img).save("lab6_recon.png")
Image.fromarray(U_vis).save("lab6_spectrum.png")

print("\nSaved files:")
print(" - lab6_orig.png")
print(" - lab6_recon.png")
print(" - lab6_spectrum.png")
