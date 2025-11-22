import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from collections import Counter

# 1. Завантаження Fashion-MNIST
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Нормалізація та перетворення у вектори
X_train = train_images.reshape(len(train_images), -1) / 255.0
X_test = test_images.reshape(len(test_images), -1) / 255.0

y_train = train_labels
y_test = test_labels

subset = 10000  # або 5000, або 8000
X_train_small = X_train[:subset]
y_train_small = y_train[:subset]


# 2. Gaussian Mixture Model — 10 компонентів
gmm = GaussianMixture(n_components=10, covariance_type="diag", random_state=42)

print("Навчаю GMM, зачекайте...")
gmm.fit(X_train)

# 3. Передбачення компонентів для тестових зображень
test_components = gmm.predict(X_test)

# 4. Призначення кожному компоненту “справжньої мітки”
component_to_label = {}

for comp in range(10):
    idxs = np.where(test_components == comp)[0]
    if len(idxs) == 0:
        component_to_label[comp] = -1
        continue
    most_common = Counter(y_test[idxs]).most_common(1)[0][0]
    component_to_label[comp] = most_common

# Прогноз (класифікація)
y_pred = np.array([component_to_label[c] for c in test_components])

# 5. Точність
acc = accuracy_score(y_test, y_pred)
print("GMM accuracy:", acc)

# 6. Візуалізація середніх значень кожного компонента
means = gmm.means_

plt.figure(figsize=(12, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(means[i].reshape(28, 28), cmap="gray")
    plt.title(f"Component {i}\nLabel {component_to_label[i]}")
    plt.axis("off")

plt.tight_layout()
plt.show()
