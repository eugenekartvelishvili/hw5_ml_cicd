import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from pathlib import Path


def load_and_split_data(test_size: float = 0.3, random_state: int = 42):
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    # Целевая колонка называется 'target'
    df["target"] = data.target

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["target"]
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame, data_dir: str = "data"):
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(data_path / "train.csv", index=False)
    test_df.to_csv(data_path / "test.csv", index=False)


if __name__ == "__main__":
    train, test = load_and_split_data()
    save_data(train, test)
    print("Данные сохранены в папку data/")