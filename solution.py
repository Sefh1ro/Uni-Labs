import io
from PIL import Image

img = Image.open("img1.jpg")
# Виводимо зображення на екран:
img.show()
print(img.size, img.format, img.mode)
print(img.info)
print(img.getbbox())

img = img.convert("L")
img.show()

# Відкриваємо файл у бінарному режимі
with open("img2.gif", "rb") as f:
    img = Image.open(f)
    img.show()


# Читаємо ВЕСЬ вміст файлу в одну змінну у вигляді байтів
with open("img2.gif", "rb") as f:
    file_bytes = f.read()

# Створюємо "файл у пам'яті" з цих байтів
img_stream = io.BytesIO(file_bytes)

# Передаємо цей об'єкт "файлу в пам'яті"
img = Image.open(img_stream)
img.show()
