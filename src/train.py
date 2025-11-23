import mlflow
import mlflow.sklearn
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from data import load_and_split_data


def train_model(
    n_estimators: int = 100,
    max_depth: int = 5,
    random_state: int = 42,
):
    # Загружаем данные
    train_df, test_df = load_and_split_data(test_size=0.3, random_state=random_state)

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]

    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    # Модель
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    # Предсказания и метрики
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return model, acc, f1, (X_train, y_train, X_test, y_test)


def main():
    # Настраиваем MLflow
    # Локальное файловое хранилище в папке mlruns
    tracking_uri = "file:./mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("hw5_breast_cancer_experiment")

    Path("artifacts").mkdir(exist_ok=True)

    with mlflow.start_run():
        n_estimators = 120
        max_depth = 6
        random_state = 42

        model, acc, f1, data = train_model(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )

        # Логируем параметры
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)

        # Логируем метрики
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # Сохраняем модель как артефакт
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Можно сохранить и сами данные
        X_train, y_train, X_test, y_test = data
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        train_path = "artifacts/train_used.csv"
        test_path = "artifacts/test_used.csv"
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        mlflow.log_artifact(train_path)
        mlflow.log_artifact(test_path)

        print(f"Модель обучена. Accuracy={acc:.4f}, F1={f1:.4f}")
        print("Логи эксперимента сохранены в mlruns/")


if __name__ == "__main__":
    main()