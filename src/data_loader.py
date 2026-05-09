"""Data loading helpers."""

import pandas as pd

__all__ = ["load_data"]


def load_data(path: str) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    return pd.read_csv(path)