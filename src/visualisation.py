import matplotlib.pyplot as plt




def plot_elbow_method(k_values, inertials):
    plt.figure(figsize=(8,5))

    plt.plot(k_values, inertials, marker='o')

    plt.title("Elbow Method")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")

    plt.grid(True)
    plt.savefig("images/elbow_method.png")
    plt.close()

def plot_clusters_pca(X_pca, label):
    plt.figure(figsize=(8,5))


    scatter = plt.scatter(
        X_pca[:, 0], #первый столбец в numpy array
        X_pca[:, 1],
        c=label,
        s=5,
        alpha=0.6, #прозрачность
    )

    plt.title("Cutsomer Clusters PCA Visualisation")
    plt.xlabel("Principial Component 1")
    plt.ylabel("Principial Component 2")
    plt.colorbar(scatter, label="Cluster")
    plt.grid(True)

    plt.savefig("images/clusters_pca.png")
    plt.close()


def plot_income_premium_clusters(df):
    plt.figure(figsize=(10,6))

    scatter = plt.scatter(
        df["Income Level"],
        df["Premium Amount"],
        c = df["Cluster"],
        s=5,
        alpha=0.6,
    )

    plt.title("Customer Clusters by Income and Premium")
    plt.xlabel("Income Level")
    plt.ylabel("Premium Amount")
    plt.colorbar(scatter, label="Cluster")
    plt.grid(True)

    plt.savefig("images/income_premium_clusters.png")
    plt.close()

