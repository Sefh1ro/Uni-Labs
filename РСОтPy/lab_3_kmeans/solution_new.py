import numpy as np
from sklearn.cluster import KMeans, MeanShift
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from skimage.feature import hog
from keras.datasets import mnist
import matplotlib.pyplot as plt


(X_train, y_train), (X_test, y_test) = mnist.load_data()
X = np.concatenate([X_train, X_test])
y = np.concatenate([y_train, y_test])

X = X[:5000]
y = y[:5000]

X_flat = X.reshape((len(X), -1))

kmeans_raw = KMeans(n_clusters=10, random_state=42)
clusters_raw = kmeans_raw.fit_predict(X_flat)

fig, axes = plt.subplots(2, 5, figsize=(8,4))
for i, ax in enumerate(axes.flat):
    ax.imshow(kmeans_raw.cluster_centers_[i].reshape(28,28), cmap='gray')
    ax.set_title(f"Cluster {i}")
    ax.axis("off")
plt.show()

def extract_hog_batch(images):
    hog_features = []
    for img in images:
        feat = hog(img, orientations=8, pixels_per_cell=(4,4),
                   cells_per_block=(2,2), visualize=False)
        hog_features.append(feat)
    return np.array(hog_features)

X_hog = extract_hog_batch(X)

kmeans_hog = KMeans(n_clusters=10, random_state=42)
clusters_hog = kmeans_hog.fit_predict(X_hog)

pca = PCA(n_components=50)   # можна підібрати оптимальне k
X_pca = pca.fit_transform(X_hog)

kmeans_pca = KMeans(n_clusters=10, random_state=42)
clusters_pca = kmeans_pca.fit_predict(X_pca)

def clustering_accuracy(true_labels, cluster_labels):
    correct = 0
    for c in np.unique(cluster_labels):
        mask = (cluster_labels == c)
        true_for_cluster = true_labels[mask]
        most_common = np.bincount(true_for_cluster).argmax()
        correct += sum(true_for_cluster == most_common)
    return correct / len(true_labels)

print("Accuracy raw:", clustering_accuracy(y, clusters_raw))
print("Accuracy hog:", clustering_accuracy(y, clusters_hog))
print("Accuracy pca:", clustering_accuracy(y, clusters_pca))

ms = MeanShift()
ms.fit(X_flat[:1000])
clusters_ms = ms.labels_
centers_ms = ms.cluster_centers_
print("Found clusters:", len(centers_ms))

fig, axes = plt.subplots(1, len(centers_ms), figsize=(15,3))
for i, ax in enumerate(axes):
    ax.imshow(centers_ms[i].reshape(28,28), cmap='gray')
    ax.set_title(f"C{i}")
    ax.axis("off")
plt.show()