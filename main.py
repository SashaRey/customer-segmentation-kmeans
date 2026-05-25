from src.data_loader import load_data
from src.report import print_basic_info
from src.preprocessing import prepare_features
from src.clustering import calculate_inertia, train_kmeans, find_optimal_k
from src.visualisation import (
    plot_elbow_method,
    plot_clusters_pca,
    plot_income_premium_clusters,
)
from src.preprocessing import apply_pca

def main():
    df = load_data("data/customer_segmentation_data.csv")
    print_basic_info(df)

    #Распакуем таблицы
    X, X_scaled = prepare_features(df)

    print("\nВыбранные признаки:")
    print(X.head())

    print("\nРазмер масштабированных данных:")
    print(X_scaled.shape)

    k_values, inertias = calculate_inertia(X_scaled)
    print("\nElbow method results")
    for k, inertia in zip(k_values, inertias):
        print(f"k=[{k}: inertia={inertia:.2f}]")
    
    plot_elbow_method(k_values, inertias)
    
    print("\nElbow method plot saved to images/elbow_method.png")

    optimal_k, distances = find_optimal_k(k_values, inertias)
    print(f"Оптимальное поличество кластеров по Kneedle: {optimal_k}")
    
    labels = train_kmeans(X_scaled, n_clusters=optimal_k)
    df["Cluster"] = labels
    

    print("\n--- Бизнес-профили кластеров (средние значения) ---")
    cluster_profiles = df.groupby("Cluster")[[
        "Age", 
        "Income Level", 
        "Coverage Amount", 
        "Premium Amount"
    ]].mean().round(2)
    print(cluster_profiles)

    # Вывод демографии (Процентное соотношение)
    print("\n--- Демография кластеров (Пол в %) ---")
    gender_pct = df.groupby("Cluster")["Gender"].value_counts(normalize=True).unstack().fillna(0) * 100
    print(gender_pct.round(1))

    print("\n--- Демография кластеров (Семейное положение в %) ---")
    status_pct = df.groupby("Cluster")["Marital Status"].value_counts(normalize=True).unstack().fillna(0) * 100
    print(status_pct.round(1))

    df.to_csv("data/customers_with_clusters.csv", index=False)

    print("\nФайл сохранён: data/customers_with_clusters.csv")

    X_pca = apply_pca(X_scaled, n_components=2)
    plot_clusters_pca(X_pca, df["Cluster"])
    print("\nPCA-график сохранён: images/clusters_pca.png")


    plot_income_premium_clusters(df)
    print("Income/Premium cluster plot сохранён: images/income_premium_clusters.png")


    

    



if __name__ == "__main__":
    main()
