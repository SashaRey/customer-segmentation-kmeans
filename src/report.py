"""Reporting helpers."""

# __all__ = ["print_basic_info"]


def print_basic_info(df):
    print("Размеры датасета:", df.shape)
    print("\nПервые строки:")
    print(df.head())

    print("\nПропуски:")
    print(df.isnull().sum())

    print("\nСтатистика:")
    print(df.describe())