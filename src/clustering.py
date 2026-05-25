from sklearn.cluster import KMeans
import numpy as np


def calculate_inertia(X_scaled):
    k_values = range(2, 15)
    inertias = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    
    return list(k_values), inertias
        

def train_kmeans(X_scaled, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_scaled) #массив номеров кластеров

    return labels


def find_optimal_k(k_values, inertias):
    """
    Метод максимального расстояния до секущей.
    """
    # Координаты первой и последней точек графика
    x1, y1 = k_values[0], inertias[0]
    x2, y2 = k_values[-1], inertias[-1]
    
    distances = []
    
    for i in range(len(k_values)):
        x0 = k_values[i]
        y0 = inertias[i]
        
        # Вычисляем расстояние от точки кривой до прямой по формуле
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distance = numerator / denominator
        distances.append(distance)
        
    # Находим индекс максимального расстояния
    optimal_index = np.argmax(distances)
    optimal_k = k_values[optimal_index]
    
    return optimal_k, distances