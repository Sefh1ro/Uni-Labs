#!/usr/bin/env python3
from PIL import Image, ImageDraw, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
import time
import math

W, H = 700, 500


def draw_point(draw, x, y, color=(255,0,0), r=3):
    draw.ellipse((x-r, y-r, x+r, y+r), fill=color)


# ---------------- Bresenham ----------------
def bresenham(img, x0,y0,x1,y1, color):
    x0,y0,x1,y1 = map(int, map(round, (x0,y0,x1,y1)))
    dx = abs(x1-x0); dy = abs(y1-y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx-dy
    while True:
        if 0<=x0<img.width and 0<=y0<img.height:
            img.putpixel((x0,y0), color)
        if x0==x1 and y0==y1: break
        e2 = 2*err
        if e2 > -dy: err -= dy; x0 += sx
        if e2 < dx:  err += dx; y0 += sy

# ---------------- Scanline fill ----------------
def scanline_fill(img, contours, fill_color):
    t0 = time.perf_counter()
    edges = []
    for poly in contours:
        for i in range(len(poly)):
            x0,y0 = poly[i]
            x1,y1 = poly[(i+1)%len(poly)]
            if y0==y1: continue
            if y0>y1: x0,y0,x1,y1 = x1,y1,x0,y0
            edges.append((x0,y0,x1,y1))
    filled = 0
    w,h = img.size
    for y in range(h):
        xs = []
        for x0,y0,x1,y1 in edges:
            if y0<=y<y1:
                t = (y-y0)/(y1-y0)
                xs.append(x0+t*(x1-x0))
        xs.sort()
        for i in range(0,len(xs),2):
            for x in range(int(xs[i]), int(xs[i+1])):
                if 0<=x<w:
                    img.putpixel((x,y), fill_color)
                    filled += 1
    return (time.perf_counter()-t0)*1000, filled

# ---------------- Cohen–Sutherland ----------------
INSIDE,LEFT,RIGHT,TOP,BOTTOM = 0,1,2,4,8
def outcode(x,y,xmin,ymin,xmax,ymax):
    c=INSIDE
    if x<xmin:c|=LEFT
    if x>xmax:c|=RIGHT
    if y<ymin:c|=TOP
    if y>ymax:c|=BOTTOM
    return c

def clip(x0,y0,x1,y1,xmin,ymin,xmax,ymax):
    c0,c1 = outcode(x0,y0,xmin,ymin,xmax,ymax), outcode(x1,y1,xmin,ymin,xmax,ymax)
    while True:
        if not (c0|c1): return x0,y0,x1,y1
        if c0&c1: return None
        c = c0 or c1
        if c&TOP:
            x = x0+(x1-x0)*(ymin-y0)/(y1-y0); y=ymin
        elif c&BOTTOM:
            x = x0+(x1-x0)*(ymax-y0)/(y1-y0); y=ymax
        elif c&LEFT:
            y = y0+(y1-y0)*(xmin-x0)/(x1-x0); x=xmin
        else:
            y = y0+(y1-y0)*(xmax-x0)/(x1-x0); x=xmax
        if c==c0: x0,y0,c0 = x,y,outcode(x,y,xmin,ymin,xmax,ymax)
        else: x1,y1,c1 = x,y,outcode(x,y,xmin,ymin,xmax,ymax)

# ---------------- App ----------------
class App:
    def __init__(self,root):
        self.root=root
        self.root.title("ЛР-2: Графічні алгоритми")
        self.canvas = tk.Canvas(root,width=W,height=H,bg="white")
        self.canvas.pack()
        self.base=None
        self.draw=None
        self.clicks=[]
        top=tk.Frame(root); top.pack()
        for t,f in [("1. Відрізок",self.mode_line),
                    ("2. Заповнення",self.mode_fill),
                    ("3. Відсічення",self.mode_clip)]:
            tk.Button(top,text=t,command=f).pack(side="left")
        tk.Button(top,text="Save",command=self.save).pack(side="left")
        tk.Button(top,text="Reset",command=self.reset).pack(side="left")
        self.mode_line()

    def reset(self):
        self.clicks=[]
        self.base=Image.new("RGB",(W,H),(255,255,255))
        self.draw=ImageDraw.Draw(self.base)
        self.update()

    def update(self):
        self.tkimg=ImageTk.PhotoImage(self.base)
        self.canvas.create_image(0,0,anchor="nw",image=self.tkimg)

    def save(self):
        path=filedialog.asksaveasfilename(defaultextension=".png")
        if path:self.base.save(path)

    # ----- Mode 1 -----
    def mode_line(self):
        self.reset()
        self.canvas.bind("<Button-1>",self.line_click)

    def line_click(self, e):
        self.clicks.append((e.x, e.y))
        draw_point(self.draw, e.x, e.y, (255, 0, 0))  # точка кліку
        if len(self.clicks) == 2:
            x0, y0 = self.clicks[0]
            x1, y1 = self.clicks[1]
            bresenham(self.base, x0, y0, x1, y1, (0, 0, 0))
            self.update()
            self.clicks = []

    # ----- Mode 2 -----
    def mode_fill(self):
        self.reset()
        simple=[[(100,100),(200,80),(300,120),(250,200),(150,200)]]
        complex_=[[(350,80),(600,120),(550,200),(400,220),(320,150)],
                  [(430,140),(460,145),(450,170),(420,160)]]
        t1,f1 = scanline_fill(self.base,simple,(200,230,255))
        t2,f2 = scanline_fill(self.base,complex_,(180,210,180))
        for p in simple+complex_:
            self.draw.line(p+[p[0]],fill=(0,0,0))
        self.update()
        print(f"Простий: {t1:.2f} ms, pixels={f1}")
        print(f"Складний: {t2:.2f} ms, pixels={f2}")

    # ----- Mode 3 -----
    def mode_clip(self):
        self.reset()
        self.clip_rect=(200,120,550,380)
        self.draw.rectangle(self.clip_rect,outline=(0,0,0),width=2)
        self.update()
        self.canvas.bind("<Button-1>",self.clip_click)

    def clip_click(self, e):
        self.clicks.append((e.x, e.y))
        draw_point(self.draw, e.x, e.y, (255, 0, 0))  # початкові точки
        if len(self.clicks) == 2:
            x0, y0 = self.clicks[0]
            x1, y1 = self.clicks[1]

            # вихідний відрізок
            self.draw.line((x0, y0, x1, y1), fill=(160, 160, 160))

            # відсічення
            r = clip(x0, y0, x1, y1, *self.clip_rect)
            if r:
                cx0, cy0, cx1, cy1 = r
                bresenham(self.base, cx0, cy0, cx1, cy1, (255, 0, 255))
                draw_point(self.draw, cx0, cy0, (255, 0, 255))
                draw_point(self.draw, cx1, cy1, (255, 0, 255))

            self.update()
            self.clicks = []


root=tk.Tk()
App(root)
root.mainloop()
