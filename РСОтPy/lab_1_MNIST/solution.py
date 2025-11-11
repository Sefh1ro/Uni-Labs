from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml
from keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
print(X.shape, y.shape)


(train_X, train_y), (test_X, test_y) = mnist.load_data()

print('X_train:', train_X.shape)
print('Y_train:', train_y.shape)
print('X_test:', test_X.shape)
print('Y_test:', test_y.shape)

for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(train_X[i], cmap=plt.get_cmap('gray'))
plt.show()

# Перетворюємо вектори
train_X_flat = train_X.reshape((train_X.shape[0], -1))
test_X_flat = test_X.reshape((test_X.shape[0], -1))

# Нормалізація (робимо значення від 0 до 1)
train_X_flat = train_X_flat / 255.0
test_X_flat = test_X_flat / 255.0

model = LogisticRegression(max_iter=1000)
model.fit(train_X_flat, train_y)

accuracy = model.score(test_X_flat, test_y)
print(f"Accuracy: {accuracy:.4f}")

for i in range(10):
    coef_image = model.coef_[i].reshape(28, 28)
    plt.subplot(2, 5, i + 1)
    plt.imshow(coef_image, cmap='seismic')
    plt.title(f'Цифра {i}')
    plt.axis('off')

plt.tight_layout()
plt.show()