from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# __all__ = ["prepare_features"]

def prepare_features(df):
    features = [
        "Age",
        "Income Level",
        "Coverage Amount",
        "Premium Amount",
    ]

    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X, X_scaled

def apply_pca(X_scaled, n_components=2):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    return X_pca
