# clustering_mnist.py — примерный скрипт
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from skimage.feature import hog
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import euclidean_distances
from collections import Counter
from tqdm import tqdm

# 0. Загрузка и простая предобработка
(train_X, train_y), (test_X, test_y) = mnist.load_data()
X = np.concatenate([train_X, test_X], axis=0)
y = np.concatenate([train_y, test_y], axis=0)
X = X.astype(np.float32) / 255.0
n_samples = 10000   # для быстроты / теста
X = X[:n_samples]
y = y[:n_samples]

# Удобная функция для визуализации ряда изображений
def show_images(img_list, titles=None, cols=5, cmap='gray'):
    rows = (len(img_list) + cols - 1) // cols
    plt.figure(figsize=(cols*2, rows*2))
    for i, img in enumerate(img_list):
        plt.subplot(rows, cols, i+1)
        plt.imshow(img, cmap=cmap)
        if titles:
            plt.title(titles[i], fontsize=8)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# 1) RAW pixels
X_raw = X.reshape(len(X), -1)  # shape (n_samples, 784)

# 2) HOG features (пример параметров)
def extract_hog(images, max_samples=None):
    feats = []
    it = images if max_samples is None else images[:max_samples]
    for img in tqdm(it, desc="HOG"):
        feats.append(hog(img, pixels_per_cell=(4,4), cells_per_block=(2,2)))
    return np.array(feats)

# Пример: считаем HOG для 5000 образов (можно меньше/больше)
X_hog = extract_hog(X, max_samples=len(X))

# 3) Top-k через PCA (уменьшаем размерность HOG или raw)
k = 50  # пробуй 20,50,100 — подобрать сам
pca = PCA(n_components=k, random_state=42)
X_pca = pca.fit_transform(X_hog)  # можно пробовать pca.fit_transform(X_raw) тоже

# ---- Функция: кластеризация + визуализация центров ----
def cluster_and_eval(X_features, how='raw_pixels', n_clusters=10):
    print(f"\n--- KMeans on {how} ---")
    km = KMeans(n_clusters=n_clusters, random_state=42)
    labels = km.fit_predict(X_features)
    centers = km.cluster_centers_

    # Визуализируем центры:
    if how == 'raw_pixels':
        imgs = [centers[i].reshape(28,28) for i in range(n_clusters)]
        show_images(imgs, titles=[f"cluster {i}" for i in range(n_clusters)], cols=5)
    else:
        # для HOG / PCA: покажем ближайшее реальное изображение к каждому центру
        dists = euclidean_distances(centers, X_features)
        nearest_idxs = dists.argmin(axis=1)
        imgs = [X[idx].reshape(28,28) for idx in nearest_idxs]
        show_images(imgs, titles=[f"cluster {i}" for i in range(n_clusters)], cols=5)

    # mapping cluster -> majority true label
    mapping = {}
    for cl in range(n_clusters):
        idxs = np.where(labels == cl)[0]
        if len(idxs) == 0:
            mapping[cl] = -1
            continue
        most_common = Counter(y[idxs]).most_common(1)[0][0]
        mapping[cl] = most_common

    # вычислим, сколько правильно
    pred_digits = np.array([mapping[l] for l in labels])
    correct = (pred_digits == y)
    acc = correct.sum() / len(y)
    print(f"KMeans-estimated accuracy (by mapping clusters->labels): {acc:.4f} ({correct.sum()} / {len(y)})")

    return km, labels, mapping

# Прогоним KMeans для трёх представлений:
km_raw, labels_raw, map_raw = cluster_and_eval(X_raw, how='raw_pixels', n_clusters=10)
km_hog, labels_hog, map_hog = cluster_and_eval(X_hog, how='hog', n_clusters=10)
km_pca, labels_pca, map_pca = cluster_and_eval(X_pca, how=f'PCA(k={k})', n_clusters=10)

# ---- MeanShift ----
print("\n--- MeanShift (на HOG, как пример) ---")
bandwidth = estimate_bandwidth(X_hog, quantile=0.2, n_samples=1000)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
labels_ms = ms.fit_predict(X_hog)
n_clusters_ms = len(np.unique(labels_ms))
print("MeanShift found clusters:", n_clusters_ms)

# Визуализация центров MeanShift (покажем ближайшие реальные образцы)
centers_ms = ms.cluster_centers_
dists = euclidean_distances(centers_ms, X_hog)
nearest = dists.argmin(axis=1)
imgs = [X[idx].reshape(28,28) for idx in nearest]
show_images(imgs, titles=[f"MS cluster {i}" for i in range(len(nearest))], cols=5)

# Оценка accuracy аналогично:
mapping_ms = {}
for cl in np.unique(labels_ms):
    idxs = np.where(labels_ms == cl)[0]
    mapping_ms[cl] = Counter(y[idxs]).most_common(1)[0][0]
pred_digits_ms = np.array([mapping_ms[l] for l in labels_ms])
acc_ms = (pred_digits_ms == y).sum() / len(y)
print(f"MeanShift (HOG) mapped accuracy: {acc_ms:.4f}")

# ---- Примеры ошибок ----
def show_errors(labels_pred, mapping):
    pred_digits = np.array([mapping[l] for l in labels_pred])
    wrong_idx = np.where(pred_digits != y)[0]
    print("Примеры ошибок, первых 12:")
    if len(wrong_idx)==0:
        print("Нет ошибок")
        return
    imgs = [X[i].reshape(28,28) for i in wrong_idx[:12]]
    titles = [f"true={y[i]} pred={pred_digits[i]}" for i in wrong_idx[:12]]
    show_images(imgs, titles=titles, cols=4)

show_errors(labels_hog, map_hog)
