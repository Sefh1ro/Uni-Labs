#!/usr/bin/env python3
"""
geom_transform_pil_tk.py
Интерактивный инструмент (Tkinter + Pillow) для affine преобразований
(масштаб, поворот, сдвиг) через 3x3 однорідну матрицю.
"""
import sys, os
from math import cos, sin, radians
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox

# ---------------- helpers for homogeneous matrices ----------------
def T(tx, ty):
    return np.array([[1,0,tx],[0,1,ty],[0,0,1]], dtype=np.float64)

def S(sx, sy):
    return np.array([[sx,0,0],[0,sy,0],[0,0,1]], dtype=np.float64)

def R(theta_rad):
    c = cos(theta_rad); s = sin(theta_rad)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float64)

def build_H(sx, sy, angle_deg, tx, ty, center):
    cx, cy = center
    theta = radians(angle_deg)
    H = T(tx, ty) @ T(cx, cy) @ R(theta) @ S(sx, sy) @ T(-cx, -cy)
    return H

# ---------------- load image ----------------
def choose_path_dialog():
    return filedialog.askopenfilename(title="Select an image",
                                      filetypes=[("Images","*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"),("All files","*.*")])

if len(sys.argv) >= 2:
    img_path = sys.argv[1]
else:
    img_path = None

root = tk.Tk()
root.withdraw()
if not img_path:
    img_path = choose_path_dialog()
root.deiconify()

if not img_path or not os.path.exists(img_path):
    messagebox.showerror("Error", "Image not provided or not found.")
    sys.exit(1)

orig = Image.open(img_path).convert("RGBA")
w, h = orig.size
center = (w/2.0, h/2.0)

# ---------------- UI ----------------
root.title("PIL / Tkinter Geometric Transform")

canvas = tk.Canvas(root, width=w, height=h)
canvas.grid(row=0, column=0, columnspan=6)

# Scale widgets
sx_var = tk.DoubleVar(value=1.0)
sy_var = tk.DoubleVar(value=1.0)
angle_var = tk.DoubleVar(value=0.0)
tx_var = tk.DoubleVar(value=0.0)
ty_var = tk.DoubleVar(value=0.0)

def on_change(_=None):
    sx = sx_var.get(); sy = sy_var.get()
    angle = angle_var.get()
    tx = tx_var.get(); ty = ty_var.get()
    H = build_H(sx, sy, angle, tx, ty, center)
    # PIL expects coefficients for inverse mapping: compute inverse H
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return
    coeffs = H_inv[:2,:].reshape(6)  # a,b,c,d,e,f
    # PIL.transform expects a tuple of 6 numbers
    transformed = orig.transform((w,h), Image.AFFINE, data=tuple(coeffs), resample=Image.BICUBIC)
    # convert to ImageTk and display
    tk_img = ImageTk.PhotoImage(transformed)
    canvas.image_ref = tk_img
    canvas.create_image(0,0,anchor='nw',image=tk_img)

# Controls layout
tk.Label(root, text="Scale X").grid(row=1, column=0)
tk.Scale(root, variable=sx_var, from_=0.1, to=3.0, resolution=0.01, orient='horizontal', length=200, command=on_change).grid(row=1, column=1)

tk.Label(root, text="Scale Y").grid(row=2, column=0)
tk.Scale(root, variable=sy_var, from_=0.1, to=3.0, resolution=0.01, orient='horizontal', length=200, command=on_change).grid(row=2, column=1)

tk.Label(root, text="Angle (deg)").grid(row=1, column=2)
tk.Scale(root, variable=angle_var, from_=-180, to=180, resolution=1, orient='horizontal', length=200, command=on_change).grid(row=1, column=3)

tk.Label(root, text="Trans X (px)").grid(row=2, column=2)
tk.Scale(root, variable=tx_var, from_=-w, to=w, resolution=1, orient='horizontal', length=200, command=on_change).grid(row=2, column=3)

tk.Label(root, text="Trans Y (px)").grid(row=3, column=2)
tk.Scale(root, variable=ty_var, from_=-h, to=h, resolution=1, orient='horizontal', length=200, command=on_change).grid(row=3, column=3)

def save_image():
    sx = sx_var.get(); sy = sy_var.get()
    angle = angle_var.get()
    tx = tx_var.get(); ty = ty_var.get()
    H = build_H(sx, sy, angle, tx, ty, center)
    H_inv = np.linalg.inv(H)
    coeffs = H_inv[:2,:].reshape(6)
    transformed = orig.transform((w,h), Image.AFFINE, data=tuple(coeffs), resample=Image.BICUBIC)
    save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                             filetypes=[("PNG","*.png"),("JPEG","*.jpg;*.jpeg"),("All files","*.*")])
    if save_path:
        transformed.save(save_path)
        messagebox.showinfo("Saved", f"Saved to: {save_path}")

tk.Button(root, text="Save...", command=save_image).grid(row=4, column=1)
tk.Button(root, text="Reset", command=lambda: (sx_var.set(1.0), sy_var.set(1.0), angle_var.set(0.0), tx_var.set(0.0), ty_var.set(0.0), on_change())).grid(row=4, column=2)
tk.Button(root, text="Quit", command=root.quit).grid(row=4, column=3)

# Initial draw
on_change()
root.mainloop()
