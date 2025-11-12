import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
from skimage.feature import hog
from keras.datasets import mnist
from tqdm import tqdm  # для зручного відображення прогресу

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Перетворення зображень у вектори
X_train_flat = X_train.reshape(len(X_train), -1)
X_test_flat = X_test.reshape(len(X_test), -1)

# Перетворення міток у бінарний формат: "5" = 1, інші = 0
y_train_bin = (y_train == 5).astype(int)
y_test_bin = (y_test == 5).astype(int)

# Створення та навчання моделі
clf_raw = svm.LinearSVC(max_iter=2000)
clf_raw.fit(X_train_flat[:10000], y_train_bin[:10000])  # для швидкості

# Перевірка якості
y_pred_raw = clf_raw.predict(X_test_flat[:2000])
print("Accuracy (без ознак):", accuracy_score(y_test_bin[:2000], y_pred_raw))

#======================================================

# Функція для обчислення HOG для набору зображень
def extract_hog_features(images):
    features = []
    for img in tqdm(images):
        hog_feature = hog(img, pixels_per_cell=(4, 4), cells_per_block=(2, 2))
        features.append(hog_feature)
    return np.array(features)

# Обчислення ознак
X_train_hog = extract_hog_features(X_train[:2000])
X_test_hog = extract_hog_features(X_test[:500])

# Навчання SVM на HOG-ознаках
clf_hog = svm.LinearSVC(max_iter=3000)
clf_hog.fit(X_train_hog, y_train_bin[:2000])

# Перевірка результату
y_pred_hog = clf_hog.predict(X_test_hog)
print("Accuracy (з HOG):", accuracy_score(y_test_bin[:500], y_pred_hog))

print("\nClassification report (з HOG):")
print(classification_report(y_test_bin[:500], y_pred_hog, target_names=["інші", "п'ятірка"]))


plt.figure(figsize=(10, 6))
for i in range(10):
    idx = np.random.randint(0, len(X_test_hog))
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')

    label_true = "п'ятірка" if y_test_bin[idx] == 1 else "інша"
    label_pred = "п'ятірка" if y_pred_hog[idx] == 1 else "інша"

    plt.title(f"Прогноз: {label_pred}\nІстинна: {label_true}")
    plt.axis('off')

plt.tight_layout()
plt.show()

