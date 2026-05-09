from src.data_loader import load_data
from src.report import print_basic_info
from src.preprocessing import prepare_features
from src.clustering import calculate_inertia
from src.visualisation import plot_elbow_method

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



if __name__ == "__main__":
    main()
