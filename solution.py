from PIL import Image

img = Image.open("img1.jpg")
# Виводимо зображення на екран:
img.show()
print(img.size, img.format, img.mode)
print(img.info)
print(img.getbbox())

# Відкриваємо файл у бінарному режимі
f = open("img2.gif", "rb")
# Передаємо об'єкт файлу
img = Image.open(f)
# Виводимо передане зображення на екран:
img.show()
f.close()
# Закриваємо файл