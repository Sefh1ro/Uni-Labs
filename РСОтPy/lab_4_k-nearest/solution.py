import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist, fashion_mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def run_knn_dataset(load_fn, dataset_name):
    print(f"\n===== KNN classification on {dataset_name} =====")

    # 1. Завантажуємо дані
    (X_train_full, y_train_full), (X_test_full, y_test_full) = load_fn()
    X = np.concatenate([X_train_full, X_test_full], axis=0)
    y = np.concatenate([y_train_full, y_test_full], axis=0)

    # нормалізація та перетворення у вектори
    X = X.astype(np.float32) / 255.0
    X = X.reshape(len(X), -1)

    # 2. Розділяємо на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.40, random_state=42
    )

    # 3. Масштабування
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # 4. Навчання kNN
    classifier = KNeighborsClassifier(n_neighbors=8)
    classifier.fit(X_train, y_train)

    # 5. Класифікація
    y_pred = classifier.predict(X_test)

    # 6. Оцінка
    acc = accuracy_score(y_test, y_pred)
    correct = (y_pred == y_test).sum()
    wrong = (y_pred != y_test).sum()

    print(f"Accuracy: {acc:.4f}")
    print(f"Correct predictions: {correct}")
    print(f"Wrong predictions: {wrong}")

    # ⬅ Повертаємо те, що потрібно для показу помилок
    return X_test, y_test, y_pred

def show_wrong_predictions(X_test, y_test, y_pred, max_examples=10):
    wrong_idx = np.where(y_test != y_pred)[0]

    print(f"\nFound {len(wrong_idx)} wrong predictions. Showing first {max_examples}.\n")

    plt.figure(figsize=(10, 4))
    for i, idx in enumerate(wrong_idx[:max_examples]):
        plt.subplot(2, max_examples//2, i+1)
        img = X_test[idx].reshape(28, 28)
        plt.imshow(img, cmap="gray")
        plt.title(f"True: {y_test[idx]}\nPred: {y_pred[idx]}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# Запускаємо MNIST
mnist_X_test, mnist_y_test, mnist_pred = run_knn_dataset(mnist.load_data, "MNIST")
#show_wrong_predictions(mnist_X_test, mnist_y_test, mnist_pred)

# Запускаємо Fashion-MNIST
fmnist_X_test, fmnist_y_test, fmnist_pred = run_knn_dataset(fashion_mnist.load_data, "Fashion-MNIST")
#show_wrong_predictions(fmnist_X_test, fmnist_y_test, fmnist_pred)
