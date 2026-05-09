from sklearn.cluster import KMeans



def calculate_inertia(X_scaled):
    k_values = range(2, 15)
    inertias = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    
    return list(k_values), inertias
        
