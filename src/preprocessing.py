from sklearn.preprocessing import StandardScaler

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