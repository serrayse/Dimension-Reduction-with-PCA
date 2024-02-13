# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:20:17 2024

@author: aserrasimsek
"""

#salinas.mat dosyasının yüklenmesi ve görselleştirilmesi

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# MATLAB dosyasından veriyi yükle
mat_veri = loadmat('Salinas.mat')

# Değişken isimlerini çıkar
degisken_isimleri = mat_veri.keys()

# Veriyi içeren değişken adını bul
veri_degisken_adi = [var_adi for var_adi in degisken_isimleri if not var_adi.startswith('__')][0]

# Değişken adını kullanarak veriye eriş
veri = mat_veri[veri_degisken_adi]

# Sahnenin bir alt kümesini görselleştir
alt_kume_veri = veri[:512, :217, :]  

# Görselleştirmek için bir bant seç
gorsellenecek_bant = 100

# Görüntüyü çiz
plt.figure(figsize=(10, 10))
plt.imshow(alt_kume_veri[:, :, gorsellenecek_bant], cmap='gray')
plt.title(f"Salinas haritası - Bant {gorsellenecek_bant + 5}")
plt.colorbar()
plt.show()





#salinas.mat dosyasının PCA uygulanarak indirgenmesi ve görselleştirilmesi

import numpy as np
import matplotlib.pyplot as plt
import scipy.io

# salinas.mat dosyasını yükleme
mat_file = scipy.io.loadmat('salinas.mat')
salinas = mat_file['salinas']

# Veriyi düzleştirme
salinas_data = salinas.reshape((512 * 217, 224))

# Ortalama çıkarma
mean_vec = np.mean(salinas_data, axis=0)
salinas_data_centered = salinas_data - mean_vec

# Kovaryans matrisini hesaplama
cov_matrix = np.cov(salinas_data_centered, rowvar=False)

# Eigenvalues ve eigenvectors hesaplama
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Büyükten küçüğe sıralama
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# İlk N eigenvektörü seçme (N: küçültülecek boyut)
total_variance = np.sum(eigenvalues)
explained_variance = np.cumsum(eigenvalues) / total_variance
N = np.argmax(explained_variance >= 0.95) + 1

selected_eigenvectors = eigenvectors[:, :N]

# Veriyi yeni boyutlarda ifade etme
salinas_pca = np.dot(salinas_data_centered, selected_eigenvectors)

# Yeni boyutlu veriyi yeniden şekillendirme
salinas_pca_reshaped = salinas_pca.reshape((512, 217, N))

# PCA sonrası Salinas haritasını görselleştirme
plt.figure(figsize=(10, 10))
plt.imshow(np.sum(salinas_pca_reshaped, axis=2), cmap='gray')
plt.title(f"PCA Sonrası Salinas haritası - Bant {gorsellenecek_bant + 5}")

plt.colorbar()
plt.show()
print("N değeri:")
print(N)




#PCA ile indirgenen salinas.mat dosyasının öbeklenmesi ve salinas_gt.mat dosyasına göre hata hesaplanması

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def k_means(data, k, max_iters=100):
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    
    for _ in range(max_iters):
        distances = np.linalg.norm(data[:, np.newaxis, :] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([data[labels == j].mean(axis=0) for j in range(k)])
        if np.all(new_centroids == centroids):
            break
        centroids = new_centroids

    return labels, centroids

def load_data(file_path):
    mat_data = loadmat(file_path)
    data_variable_name = [var_name for var_name in mat_data if not var_name.startswith('__')][0]
    return mat_data[data_variable_name]

def visualize_clusters(cluster_labels, title):
    plt.imshow(cluster_labels, cmap='tab20')
    plt.title(title)
    plt.colorbar()

def plot_confusion_matrix(confusion_matrix):
    plt.figure(figsize=(12, 12))
    plt.imshow(confusion_matrix, cmap='Purples', interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.show()

def calculate_error_rate(corrected_labels, ground_truth_flat):
    error_rate = np.sum(corrected_labels != ground_truth_flat) / len(ground_truth_flat)
    print("Clustering Error Rate:", error_rate)

# İlk kodun kısmı
mat_file = loadmat('salinas.mat')
salinas = mat_file['salinas']
salinas_data = salinas.reshape((512 * 217, 224))
mean_vec = np.mean(salinas_data, axis=0)
salinas_data_centered = salinas_data - mean_vec
cov_matrix = np.cov(salinas_data_centered, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]
total_variance = np.sum(eigenvalues)
explained_variance = np.cumsum(eigenvalues) / total_variance
N = np.argmax(explained_variance >= 0.95) + 1
selected_eigenvectors = eigenvectors[:, :N]
salinas_pca = np.dot(salinas_data_centered, selected_eigenvectors)
salinas_pca_reshaped = salinas_pca.reshape((512, 217, N))

# İkinci kodun kısmı
subset_data = salinas_pca_reshaped
reshaped_data = subset_data.reshape((-1, subset_data.shape[N]))

n_clusters = 17
labels, _ = k_means(reshaped_data, n_clusters)
cluster_labels = labels.reshape(subset_data.shape[:N])

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 2)
visualize_clusters(cluster_labels, "K-Means Clustering Result")

ground_truth = load_data('Salinas_gt.mat')

ground_truth_flat = ground_truth.flatten()
cluster_labels_flat = cluster_labels.flatten()
non_zero_indices = np.logical_and(ground_truth_flat != 0, cluster_labels_flat != 0)
ground_truth_non_zero = ground_truth_flat[non_zero_indices]
cluster_labels_non_zero = cluster_labels_flat[non_zero_indices]

confusion_matrix = np.zeros((17, 17))
for i in range(len(ground_truth_non_zero)):
    confusion_matrix[ground_truth_non_zero[i], cluster_labels_non_zero[i]] += 1

plot_confusion_matrix(confusion_matrix)

best_matches = np.argmax(confusion_matrix, axis=1)
corrected_labels = np.zeros_like(cluster_labels_flat)
corrected_labels[non_zero_indices] = best_matches[ground_truth_non_zero]

calculate_error_rate(corrected_labels, ground_truth_flat)