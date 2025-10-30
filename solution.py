import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, hist, show
from PIL import Image

# завдання 1 (Заміна каналів. Методи Getpixel, putpixel)
img = Image.open('1.jpg')   #.convert('L')

for x in range(img.size[0]):
    for y in range(img.size[1]):
        r, g, b = img.getpixel((x,y))
    img.putpixel((x,y),(b, r,g))
img.show()

# завдання 2 (Складання зображення з відомих каналів зображення)
img = Image.open("1.jpg")
R, G, B = img.split()

img2 = Image.merge("RGB", (R, G, B))
print(img2.mode)

img2.show()

# завдання 3 (Побудова гістограми)
im = np.array(Image.open('1.jpg').convert('L'))

# Створюємо гістограму
plt.figure("Image histogram")
plt.hist(im.flatten(), bins=128, color='gray')
plt.title("Histogram of grayscale image")
plt.xlabel("Pixel value (0-255)")
plt.ylabel("Frequency")
plt.show()
