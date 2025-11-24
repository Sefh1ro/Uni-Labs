#!/usr/bin/env python3
"""
geom_transform_cv.py

Інтерактивний інструмент для геометричних перетворень (масштаб, поворот, зсув)
в однорідних координатах. Використовує OpenCV для GUI (trackbars) та відображення.

Запуск:
    python geom_transform_cv.py path/to/image.jpg

Керування:
 - trackbars: Scale X, Scale Y, Angle (deg), Trans X, Trans Y
 - клавіша 's' -> зберегти поточне трансформоване зображення (saved.png)
 - клавіша 'r' -> скинути параметри (поворот=0, масштаб=1, зсув=0)
 - клавіша 'q' або ESC -> вийти
"""
import sys
import os
import cv2
import numpy as np
from math import cos, sin, radians
try:
    # зручний file dialog, якщо шлях не передали
    import tkinter as tk
    from tkinter import filedialog
    TK_AVAILABLE = True
except Exception:
    TK_AVAILABLE = False

# ---- Допоміжні функції для створення однорідних матриць ----
def T(tx, ty):
    """Матриця зсуву 3x3"""
    return np.array([[1, 0, tx],
                     [0, 1, ty],
                     [0, 0, 1]], dtype=np.float64)

def S(sx, sy):
    """Матриця масштабування 3x3"""
    return np.array([[sx, 0, 0],
                     [0, sy, 0],
                     [0, 0, 1]], dtype=np.float64)

def R(theta_rad):
    """Матриця повороту (навколо початку координат) 3x3"""
    c = cos(theta_rad)
    s = sin(theta_rad)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=np.float64)

# ---- Обробка аргументів / відкриття файлу ----
def choose_image_path_from_dialog():
    if not TK_AVAILABLE:
        return None
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(title="Select image", filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"), ("All files","*.*")])
    root.destroy()
    return path

if len(sys.argv) >= 2:
    img_path = sys.argv[1]
else:
    img_path = choose_image_path_from_dialog()

if not img_path or not os.path.exists(img_path):
    print("Image not provided or file not found. Provide path as argument or enable tkinter file dialog.")
    sys.exit(1)

img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
if img is None:
    print("Не вдалося відкрити зображення:", img_path)
    sys.exit(1)

h, w = img.shape[:2]
cx, cy = w/2.0, h/2.0  # центр зображення

# ---- GUI та трекбари ----
win_name = "Geometric Transformations (homogeneous) - press 'q' to quit"
cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

# Трекбар працює з integer значеннями -> робимо масштабування до потрібних діапазонів
# Scale: 10..300 -> mapped to 0.1 .. 3.0
def nothing(x):
    pass

cv2.createTrackbar('Scale X x100', win_name, 100, 300, nothing)  # старт 100 -> 1.0
cv2.createTrackbar('Scale Y x100', win_name, 100, 300, nothing)
cv2.createTrackbar('Angle deg', win_name, 0, 360, nothing)       # 0..360
# Translation in pixels: center-based:  -w..w  we map trackbar 0..(2*w) to -w..w
cv2.createTrackbar('Trans X', win_name, w, 2*w, nothing)
cv2.createTrackbar('Trans Y', win_name, h, 2*h, nothing)

print("Instructions:")
print(" - Use trackbars to change Scale X, Scale Y, Angle, Trans X, Trans Y.")
print(" - Press 's' to save result to 'saved.png'.")
print(" - Press 'r' to reset to defaults.")
print(" - Press 'q' or ESC to exit.")

def get_trackbar_values():
    sx = cv2.getTrackbarPos('Scale X x100', win_name) / 100.0
    sy = cv2.getTrackbarPos('Scale Y x100', win_name) / 100.0
    angle = cv2.getTrackbarPos('Angle deg', win_name)
    tx = cv2.getTrackbarPos('Trans X', win_name) - w
    ty = cv2.getTrackbarPos('Trans Y', win_name) - h
    return sx, sy, angle, tx, ty

def build_combined_homogeneous(sx, sy, angle_deg, tx, ty):
    """
    Побудова комбінованої матриці H (3x3) у порядку:
      1) Перенести центр у початок координат (T_-center)
      2) Масштабування S
      3) Поворот навколо початку (R)
      4) Повернути центр назад (T_center)
      5) Застосувати зсув (T(tx,ty))
    Тобто: H = T(tx,ty) * T_center * R * S * T_-center
    Це дає зручне обертання/масштабування навколо центра зображення.
    """
    theta = radians(angle_deg)
    M_t_neg = T(-cx, -cy)
    M_s = S(sx, sy)
    M_r = R(theta)
    M_t_pos = T(cx, cy)
    M_trans = T(tx, ty)
    H = M_trans @ M_t_pos @ M_r @ M_s @ M_t_neg
    return H

def apply_transform(img, H):
    """
    Застосувати 3x3 гомогенну матрицю до зображення.
    OpenCV має warpAffine (2x3) та warpPerspective (3x3). Оскільки трансформація афінна,
    можна взяти перші два рядки H і використати warpAffine.
    (warpPerspective також працює, але для афінних перетворень 2x3 достатньо.)
    """
    H_affine = H[:2, :]
    transformed = cv2.warpAffine(img, H_affine, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return transformed

# Основний цикл
while True:
    sx, sy, angle, tx, ty = get_trackbar_values()
    H = build_combined_homogeneous(sx, sy, angle, tx, ty)
    out = apply_transform(img, H)

    # Додатково: показати матрицю на зображенні
    overlay = out.copy()
    text = f"sx={sx:.2f} sy={sy:.2f} angle={angle:.1f} tx={int(tx)} ty={int(ty)}"
    cv2.putText(overlay, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    # Також можемо вивести матрицю H
    H_flat = np.array2string(H, precision=2, suppress_small=True)
    # Для зручності не малюємо всю матрицю текстом на картинці якщо зображення маленьке
    if max(w,h) > 200:
        lines = H_flat.replace('[','').replace(']','').splitlines()
        y0 = 40
        for i, ln in enumerate(lines):
            cv2.putText(overlay, ln.strip(), (10, y0 + i*18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1, cv2.LINE_AA)

    cv2.imshow(win_name, overlay)
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q') or key == 27:
        break
    elif key == ord('s'):
        save_name = "saved.png"
        cv2.imwrite(save_name, out)
        print(f"Saved: {save_name}")
    elif key == ord('r'):
        # скидання до дефолтних значень
        cv2.setTrackbarPos('Scale X x100', win_name, 100)
        cv2.setTrackbarPos('Scale Y x100', win_name, 100)
        cv2.setTrackbarPos('Angle deg', win_name, 0)
        cv2.setTrackbarPos('Trans X', win_name, w)
        cv2.setTrackbarPos('Trans Y', win_name, h)

cv2.destroyAllWindows()
