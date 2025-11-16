import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from skimage.feature import hog
from keras.datasets import mnist
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

# Завантаження даних MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Візьмемо підвибірку для швидкості обчислень
n_samples = 5000
X_sample = X_train[:n_samples]
y_sample = y_train[:n_samples]


# Завдання 1: Кластеризація без знаходження ознак (сирі пікселі)
print("ВАРІАНТ 1: Кластеризація на сирих пікселях")

# Перетворення зображень у вектори
X_flat = X_sample.reshape(n_samples, -1)
X_flat_norm = X_flat / 255.0  # Нормалізація

# K-means кластеризація
kmeans_raw = KMeans(n_clusters=10, random_state=42, n_init=10, max_iter=300)
labels_raw = kmeans_raw.fit_predict(X_flat_norm)

# Візуалізація центрів кластерів
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('ВАРІАНТ 1: Центри кластерів (сирі пікселі)', fontsize=14, fontweight='bold')
for i, ax in enumerate(axes.flat):
    center = kmeans_raw.cluster_centers_[i].reshape(28, 28)
    ax.imshow(center, cmap='gray')
    ax.set_title(f'Кластер {i}')
    ax.axis('off')
plt.tight_layout()
plt.savefig('variant1_centers.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# ВАРІАНТ 2: Кластеризація з HOG ознаками
print("ВАРІАНТ 2: Кластеризація з HOG ознаками")

# Функція для обчислення HOG
def extract_hog_features(images):
    features = []
    for img in tqdm(images, desc="Обчислення HOG"):
        hog_feature = hog(img, pixels_per_cell=(4, 4), cells_per_block=(2, 2),
                          feature_vector=True)
        features.append(hog_feature)
    return np.array(features)


X_hog = extract_hog_features(X_sample)

# K-means на HOG ознаках
kmeans_hog = KMeans(n_clusters=10, random_state=42, n_init=10, max_iter=300)
labels_hog = kmeans_hog.fit_predict(X_hog)

# Візуалізація: показуємо приклади з кожного кластера
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('ВАРІАНТ 2: Приклади з кожного кластера (HOG)', fontsize=14, fontweight='bold')
for i, ax in enumerate(axes.flat):
    cluster_indices = np.where(labels_hog == i)[0]
    if len(cluster_indices) > 0:
        sample_idx = cluster_indices[0]
        ax.imshow(X_sample[sample_idx], cmap='gray')
        ax.set_title(f'Кластер {i}\n(справжня мітка: {y_sample[sample_idx]})')
    ax.axis('off')
plt.tight_layout()
plt.savefig('variant2_examples.png', dpi=150, bbox_inches='tight')
plt.show()


# ВАРІАНТ 3: Кластеризація з PCA (вибір k найважливіших ознак)
print("ВАРІАНТ 3: Кластеризація з PCA (відбір важливих ознак)")

# Підбір оптимального k за допомогою explained variance
pca_test = PCA(n_components=100)
pca_test.fit(X_hog)

# Знайдемо k, яке пояснює 90% дисперсії
cumsum = np.cumsum(pca_test.explained_variance_ratio_)
k_optimal = np.argmax(cumsum >= 0.90) + 1
print(f"Оптимальне k для 90% дисперсії: {k_optimal}")

# Візуалізація explained variance
# plt.figure(figsize=(10, 5))
# plt.plot(cumsum[:50], marker='o')
# plt.axhline(y=0.90, color='r', linestyle='--', label='90% дисперсії')
# plt.axvline(x=k_optimal, color='g', linestyle='--', label=f'k={k_optimal}')
# plt.xlabel('Кількість компонент')
# plt.ylabel('Кумулятивна explained variance')
# plt.title('Вибір оптимальної кількості компонент PCA')
# plt.grid(True, alpha=0.3)
# plt.legend()
# plt.savefig('pca_variance.png', dpi=150, bbox_inches='tight')
# plt.show()

# PCA з оптимальним k
pca = PCA(n_components=k_optimal)
X_pca = pca.fit_transform(X_hog)
print(f"Розмір після PCA: {X_pca.shape}")

# K-means на PCA ознаках
print("\nЗастосування K-means на PCA ознаках...")
kmeans_pca = KMeans(n_clusters=10, random_state=42, n_init=10, max_iter=300)
labels_pca = kmeans_pca.fit_predict(X_pca)

# Mean Shift для оцінки кількості кластерів
# =============================================================================
print("\n" + "=" * 70)
print("Mean Shift: Автоматична оцінка кількості кластерів")


# Використаємо підвибірку для швидкості
n_meanshift = 2000  # Збільшили для кращої оцінки
X_ms_sample = X_pca[:n_meanshift]
y_ms_sample = y_sample[:n_meanshift]



# Тестуємо різні quantile
best_quantile = None
best_n_clusters = 0

for q in [0.05, 0.1, 0.15, 0.2, 0.25]:
    bw = estimate_bandwidth(X_ms_sample, quantile=q, n_samples=500)
    ms_test = MeanShift(bandwidth=bw, bin_seeding=True)
    labels_test = ms_test.fit_predict(X_ms_sample)
    n_clust = len(np.unique(labels_test))

    # Обираємо той що дає близько 10 кластерів
    if abs(n_clust - 10) < abs(best_n_clusters - 10):
        best_quantile = q
        best_n_clusters = n_clust


# Фінальний Mean Shift
bandwidth = estimate_bandwidth(X_ms_sample, quantile=best_quantile, n_samples=500)

print("\nЗастосування Mean Shift...")
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
labels_ms = ms.fit_predict(X_ms_sample)
n_clusters_ms = len(np.unique(labels_ms))


# Візуалізація центрів Mean Shift
n_rows = (n_clusters_ms + 4) // 5
fig, axes = plt.subplots(n_rows, 5, figsize=(15, 3 * n_rows))
axes = axes.flatten() if n_clusters_ms > 5 else axes
fig.suptitle(f'Mean Shift: Знайдено {n_clusters_ms} кластерів', fontsize=14, fontweight='bold')

for i in range(len(axes)):
    ax = axes[i]
    if i < n_clusters_ms:
        cluster_indices = np.where(labels_ms == i)[0]
        if len(cluster_indices) > 0:
            # Показуємо 3 приклади з кластера
            sample_idx = cluster_indices[0]
            ax.imshow(X_sample[sample_idx], cmap='gray')

            # Статистика
            cluster_labels = y_ms_sample[cluster_indices]
            most_common = np.bincount(cluster_labels).argmax()
            count = np.sum(cluster_labels == most_common)
            ax.set_title(f'Кластер {i} (n={len(cluster_indices)})\nпереважає: {most_common} ({count})',
                         fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig('meanshift_clusters.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# Оцінка якості кластеризації
print("\n" + "=" * 70)
print("ОЦІНКА ЯКОСТІ КЛАСТЕРИЗАЦІЇ")


def evaluate_clustering(labels_pred, labels_true):
    """
    Оцінка якості кластеризації з використанням Hungarian algorithm
    для знаходження оптимального співставлення кластерів з класами
    """
    # Створюємо матрицю відповідності
    n_clusters = len(np.unique(labels_pred))
    n_classes = len(np.unique(labels_true))

    # Матриця для підрахунку
    matrix = np.zeros((n_clusters, n_classes), dtype=np.int64)

    for i in range(len(labels_pred)):
        matrix[labels_pred[i], labels_true[i]] += 1

    # Hungarian algorithm для оптимального співставлення
    row_ind, col_ind = linear_sum_assignment(-matrix)

    # Підрахунок правильних класифікацій
    correct = sum([matrix[i, j] for i, j in zip(row_ind, col_ind)])
    total = len(labels_pred)
    accuracy = correct / total

    return accuracy, correct, total, dict(zip(row_ind, col_ind))


print("\n1. ВАРІАНТ 1 (Сирі пікселі):")
acc1, correct1, total1, mapping1 = evaluate_clustering(labels_raw, y_sample)
print(f"   Точність: {acc1:.4f}")
print(f"   Правильно: {correct1}/{total1}")
print(f"   Помилково: {total1 - correct1}/{total1}")
print(f"   Співставлення кластер->клас: {mapping1}")

print("\n2. ВАРІАНТ 2 (HOG ознаки):")
acc2, correct2, total2, mapping2 = evaluate_clustering(labels_hog, y_sample)
print(f"   Точність: {acc2:.4f}")
print(f"   Правильно: {correct2}/{total2}")
print(f"   Помилково: {total2 - correct2}/{total2}")
print(f"   Співставлення кластер->клас: {mapping2}")

print("\n3. ВАРІАНТ 3 (PCA ознаки):")
acc3, correct3, total3, mapping3 = evaluate_clustering(labels_pca, y_sample)
print(f"   Точність: {acc3:.4f}")
print(f"   Правильно: {correct3}/{total3}")
print(f"   Помилково: {total3 - correct3}/{total3}")
print(f"   Співставлення кластер->клас: {mapping3}")

# Порівняльний графік
plt.figure(figsize=(10, 6))
methods = ['Сирі пікселі', 'HOG ознаки', 'PCA ознаки']
accuracies = [acc1, acc2, acc3]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

bars = plt.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black')
plt.ylabel('Точність кластеризації', fontsize=12)
plt.title('Порівняння методів кластеризації MNIST', fontsize=14, fontweight='bold')
plt.ylim([0, 1])
plt.grid(axis='y', alpha=0.3)

# Додаємо значення на стовпчики
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height,
             f'{acc:.3f}',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('comparison.png', dpi=150, bbox_inches='tight')
plt.show()


# Візуалізація помилок для найкращого методу
# best_labels = labels_hog if acc2 >= max(acc1, acc3) else (labels_pca if acc3 >= acc1 else labels_raw)
# best_mapping = mapping2 if acc2 >= max(acc1, acc3) else (mapping3 if acc3 >= acc1 else mapping1)
# best_name = "HOG" if acc2 >= max(acc1, acc3) else ("PCA" if acc3 >= acc1 else "Raw")
#
# print(f"\n\nВізуалізація помилок для найкращого методу: {best_name}")
#
# # Створюємо передбачені мітки
# predicted_labels = np.array([best_mapping.get(label, -1) for label in best_labels])
#
# # Знаходимо помилкові класифікації
# errors = predicted_labels != y_sample
# error_indices = np.where(errors)[0]
#
# if len(error_indices) > 0:
#     fig, axes = plt.subplots(3, 5, figsize=(15, 9))
#     fig.suptitle(f'Приклади помилкових класифікацій ({best_name})',
#                  fontsize=14, fontweight='bold')
#
#     for i, ax in enumerate(axes.flat):
#         if i < len(error_indices) and i < 15:
#             idx = error_indices[i]
#             ax.imshow(X_sample[idx], cmap='gray')
#             ax.set_title(f'Справжня: {y_sample[idx]}\nПрогноз: {predicted_labels[idx]}',
#                          color='red')
#         ax.axis('off')
#
#     plt.tight_layout()
#     plt.savefig('errors.png', dpi=150, bbox_inches='tight')
#     plt.show()
