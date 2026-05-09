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
