from PIL import Image

img = Image.open("foto.jpg")
print(img.mode)

#Завдання 1
pixels = img.load() # Створюємо об'єкт доступу до пікселів
# Отримати колір пікселя з координатами (25, 45)
print(f"Було: {pixels[25, 45]}")
# Встановити новий колір (червоний) для цього пікселя
pixels[25, 45] = (255, 0, 0)
print(f"Стало: {pixels[25, 45]}\n")
img.show()


# Отримати колір
print(f"Було: {img.getpixel((25, 45))}")
# Встановити колір (синій)
img.putpixel((25, 45), (0, 0, 255))
print(f"Змінили putpixel та отримали getpixel:  {img.getpixel((25, 45))}\n")
img.show()

#Завдання 2
R, G, B = img.split()
mask = Image.new("L", img.size, 128)
imgRGBA_1 = Image.merge("RGBA", (R, G, B, mask))
print(f"Через split() та merge(): {imgRGBA_1.mode}\n")
imgRGBA_1.show()

img = Image.open("foto.jpg")
print(f"{img.mode}\n")

imgRGBA_2 = img.convert("RGBA")
print(f"Через convert(): {imgRGBA_2.mode}\n")

imgRGBA_3 = img.convert("P", dither=Image.FLOYDSTEINBERG, palette=Image.ADAPTIVE, colors=128)
print(f"Конвертація в палітру ('P'): {imgRGBA_3.mode}\n")
imgRGBA_3.show()

#Завдання 3
img.save("tmp.jpg")

img.save("tmp.bmp", "BMP")

#Завдання 4
# Сучасна версія коду
img_new = Image.new("RGB", (640, 480))

# Проходимо по кожному пікселю
for x in range(640): # range замість xrange
    for y in range(480):
        # Встановлюємо колір, який залежить від координат пікселя
        r = x // 3 # Червоний залежить від позиції по горизонталі
        g = (x + y) // 6 # Зелений - від суми координат
        b = y // 3 # Синій - від позиції по вертикалі
        img_new.putpixel((x, y), (r, g, b))

img_new.show()
