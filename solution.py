from PIL import Image

img = Image.open("foto.jpg")  # Відкриваємо зображення з файлу "foto.jpg"
img2 = img.copy()  # Створюємо копію відкритого зображення
img2.show()  # Виводимо копію зображення на екран (відкриється у стандартному переглядачі)

#Завдання 1
print(f"Початкові розміри для thumbnail(): \n {img.size}")  # Виводимо початкові розміри

img.thumbnail((400, 300), Image.Resampling.LANCZOS)  # Зменшуємо зображення пропорційно, не перевищуючи (400, 300)
print(img.size)  # Виводимо нові розміри (наприклад, (400, 300))

img = Image.open("foto.jpg")  # Відкриваємо файл
print(f"Початкові розміри для resize(): \n {img.size}")  # Початкові розміри (800, 600)


newsize = (400, 400)  # Новий розмір
imgnr = img.resize(newsize)  # Створюємо нову копію з новими розмірами
imgnr.show()  # Показуємо зменшене зображення
print(imgnr.size)

#Завдання 2
img_crop_1 = img.crop([0, 0, 100, 100])  # Вирізаємо фрагмент (ліва_верхня_x, ліва_верхня_y, права_нижня_x, права_нижня_y)
img_crop_1.load()  # Завантажуємо дані фрагмента
print("==========================================")
print(img_crop_1.size)  # Виводимо розмір фрагмента (100, 100)
img_crop_1.show()

img = Image.open("tsveti.jpg")  # Відкриваємо інше зображення
print(f"Початкові розміри до crop: \n{img.size}")

box = (100, 100, 300, 300)  # Координати прямокутника
img_crop_2 = img.crop(box)  # Вирізаємо фрагмент
newsize = (400, 400)  # Новий розмір
img2nr = img_crop_2.resize(newsize)  # Масштабуємо фрагмент
img2nr.show()  # Показуємо збільшену вирізку
print(imgnr.size)

#Завдання 3
img = Image.open("foto.jpg")
print("==========================================")
print(img.size)  # (800, 600)

img2 = img.rotate(90, expand=True)  # Поворот на 90° проти годинникової стрілки
print(img2.size)  # (600, 800)
img2.show()

img3 = img.transpose(Image.FLIP_LEFT_RIGHT)  # Горизонтальне віддзеркалення
img3.show()

img4 = img.transpose(Image.FLIP_TOP_BOTTOM)  # Вертикальне віддзеркалення
img4.show()
