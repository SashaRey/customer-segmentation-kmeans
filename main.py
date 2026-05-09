from src.data_loader import load_data
from src.report import print_basic_info

def main():
    df = load_data("data/customer_segmentation_data.csv")
    print_basic_info(df)

if __name__ == "__main__":
    main()
