from PIL import Image
from pylab import *

#Завдання 1
img = Image.open("foto.jpg")  # Відкриваємо зображення
img.paste((255, 0, 0), (0, 0, 100, 100))  # Заливаємо квадрат 100×100 пікселів червоним кольором (RGB)
img.show()

img = Image.open("foto.jpg")
img.paste((0, 128, 0), img.getbbox())  # Заливаємо весь розмір зображення зеленим
img.show()

#Завдання 2
img = Image.open("foto.jpg")
img2 = img.resize((200, 150))  # Створюємо мініатюру (зменшене зображення)
print(img2.size)  # (200, 150)

img.paste((255, 0, 0), (9, 9, 211, 161))  # Малюємо червону рамку 200×150 + по 1 пікселю з кожного боку
img.paste(img2, (10, 10))  # Вставляємо мініатюру всередину рамки
img.show()


pil_im = Image.open("foto.jpg")
box = (100, 100, 400, 400)  # Координати вирізаного фрагмента
region = pil_im.crop(box)  # Вирізаємо частину
region = region.transpose(Image.ROTATE_180)  # Повертаємо фрагмент на 180 градусів
pil_im.paste(region, box)  # Вставляємо назад у те ж місце
pil_im.show()

#Завдання 3
im = array(Image.open("tsveti.jpg"))  # Перетворюємо зображення в масив NumPy
imshow(im)  # Відображаємо зображення
print("Please click 3 points")
x = ginput(3)  # Чекаємо 3 кліки мишкою на зображенні
print("You clicked:", x)

#Завдання 3
im = array(Image.open("tsveti.jpg"))  # Завантажуємо зображення у вигляді масиву
imshow(im)  # Відображаємо зображення

x = [100, 100, 400, 400]  # Координати X точок
y = [200, 500, 200, 500]  # Координати Y точок

plot(x, y, '*r')  # Малюємо червоні зірочки у цих точках
plot(x[:2], y[:2])  # З’єднуємо перші дві точки лінією
title('Plotting: "tsveti.jpg"')  # Додаємо заголовок
show()