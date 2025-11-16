import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans, MeanShift
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from skimage.feature import hog
from scipy.stats import mode

# Завантаження бази MNIST
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)

print("Розмірність X:", X.shape)
print("Приклад міток:", y[:10])

# Для прискорення роботи візьмемо лише частину даних (наприклад, 10000)
# (Повна база 70 000 — може довго працювати)
subset = 10000
X = X[:subset]
y = y[:subset]

kmeans_raw = KMeans(n_clusters=10, random_state=42)
kmeans_raw.fit(X)
labels_raw = kmeans_raw.labels_

fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(kmeans_raw.cluster_centers_[i].reshape(28, 28), cmap='gray')
    ax.set_title(f'Кластер {i}')
    ax.axis('off')
plt.suptitle("Центри кластерів (сирі дані)")
plt.show()

def cluster_accuracy(y_true, cluster_labels):
    labels_map = {}
    for i in range(10):
        mask = cluster_labels == i
        if np.any(mask):
            labels_map[i] = mode(y_true[mask]).mode[0]
    y_pred = np.array([labels_map[l] for l in cluster_labels])
    return accuracy_score(y_true, y_pred)

acc_raw = cluster_accuracy(y, labels_raw)
print(f"Точність кластеризації (сирі пікселі): {acc_raw:.3f}")
