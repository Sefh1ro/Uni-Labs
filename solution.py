from PIL import Image, ImageFilter
from pylab import *
import os


#Завадння 1
# Функція для отримання списку зображень у поточній папці
def get_imlist(path):
    """Повертає список усіх .jpg файлів у вказаній директорії"""
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]

# Отримуємо список усіх зображень у поточній папці
filelist = get_imlist(".")   # "." означає поточну директорію

# Конвертуємо файли у формат JPEG (якщо потрібно)
for infile in filelist:
    outfile = os.path.splitext(infile)[0] + ".jpg"
    if infile != outfile:
        try:
            Image.open(infile).save(outfile)
        except IOError:
            print("Cannot convert:", infile)

# Відкриваємо одне зображення і застосовуємо фільтри
img = Image.open("foto.jpg")

# приклад одного фільтра
img2 = img.filter(ImageFilter.EMBOSS)
img2.show()

# Перебираємо всі фільтри і показуємо результат
# for f in [ImageFilter.BLUR, ImageFilter.CONTOUR, ImageFilter.DETAIL,
#           ImageFilter.EDGE_ENHANCE, ImageFilter.EDGE_ENHANCE_MORE,
#           ImageFilter.EMBOSS, ImageFilter.FIND_EDGES,
#           ImageFilter.SHARPEN, ImageFilter.SMOOTH, ImageFilter.SMOOTH_MORE]:
#     img.filter(f).show()


#Завадння 2
# читаємо зображення у відтінках сірого
im = array(Image.open('foto.jpg').convert('L'))

# створюємо нову фігуру
figure()

# вимикаємо кольори
gray()

# будуємо контури
contour(im, origin='image')
axis('equal')
axis('off')
show()