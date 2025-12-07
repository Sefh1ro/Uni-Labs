import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import math

# -------------------------------------------------
# Basic setup
# -------------------------------------------------
WIDTH, HEIGHT = 600, 600

root = tk.Tk()
root.title("Procedural Graphics: Sphere & Mandelbrot")

canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="black")
canvas.pack()

image = Image.new("RGB", (WIDTH, HEIGHT), "black")
tk_image = None

# -------------------------------------------------
# Utility
# -------------------------------------------------
def update_canvas():
    global tk_image
    tk_image = ImageTk.PhotoImage(image)
    canvas.create_image(0, 0, anchor="nw", image=tk_image)

def clear():
    global image
    image = Image.new("RGB", (WIDTH, HEIGHT), "black")
    canvas.delete("all")
    update_canvas()

def save_image():
    path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")]
    )
    if path:
        image.save(path)

# -------------------------------------------------
# 1. Procedural Sphere (Gradient Filling)
# -------------------------------------------------
def draw_sphere():
    global image
    clear()

    cx, cy = WIDTH // 2, HEIGHT // 2
    R = 200
    R2 = R * R

    for x in range(-R, R + 1):
        for y in range(-R, R + 1):
            r2 = x * x + y * y
            if r2 <= R2:
                k = 1 - r2 / R2        # reflection coefficient
                intensity = int(255 * k)
                color = (intensity, intensity, 255)
                image.putpixel((cx + x, cy + y), color)

    update_canvas()

# -------------------------------------------------
# 2. Mandelbrot Fractal
# -------------------------------------------------
def draw_mandelbrot():
    global image
    clear()

    max_iter = 80
    xmin, xmax = -2.0, 1.0
    ymin, ymax = -1.5, 1.5

    for px in range(WIDTH):
        for py in range(HEIGHT):
            x0 = xmin + px / WIDTH * (xmax - xmin)
            y0 = ymin + py / HEIGHT * (ymax - ymin)

            x = y = 0.0
            iteration = 0

            while x * x + y * y <= 4 and iteration < max_iter:
                xtemp = x * x - y * y + x0
                y = 2 * x * y + y0
                x = xtemp
                iteration += 1

            color_value = int(255 * iteration / max_iter)
            image.putpixel((px, py), (0, color_value, 0))

    update_canvas()

# -------------------------------------------------
# UI Buttons
# -------------------------------------------------
frame = tk.Frame(root)
frame.pack(pady=10)

tk.Button(frame, text="Draw Sphere", command=draw_sphere, width=15).grid(row=0, column=0, padx=5)
tk.Button(frame, text="Draw Mandelbrot", command=draw_mandelbrot, width=15).grid(row=0, column=1, padx=5)
tk.Button(frame, text="Clear", command=clear, width=10).grid(row=0, column=2, padx=5)
tk.Button(frame, text="Save", command=save_image, width=10).grid(row=0, column=3, padx=5)
tk.Button(frame, text="Quit", command=root.quit, width=10).grid(row=0, column=4, padx=5)

update_canvas()
root.mainloop()
