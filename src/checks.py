from pathlib import Path

import numpy as np

# Костыль для NumPy 2.x: добавляем атрибуты, которые ждёт deepchecks
if not hasattr(np, "Inf"):
    np.Inf = np.inf
if not hasattr(np, "NINF"):
    np.NINF = -np.inf

import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import full_suite

from data import load_and_split_data
from train import train_model


def main():
    # Загружаем данные и модель
    train_df, test_df = load_and_split_data()
    model, acc, f1, data = train_model()

    X_train, y_train, X_test, y_test = data

    # Оборачиваем в deepchecks.Dataset
    train_ds = Dataset(
        X_train,
        label=y_train,
        features=X_train.columns.tolist(),
        cat_features=[],
        label_type="binary",
    )

    test_ds = Dataset(
        X_test,
        label=y_test,
        features=X_test.columns.tolist(),
        cat_features=[],
        label_type="binary",
    )

    # Полный suite
    suite = full_suite()
    result = suite.run(train_dataset=train_ds, test_dataset=test_ds, model=model)

    Path("reports").mkdir(exist_ok=True)

    # 1. Большой интерактивный HTML у меня не открывает почему-то 
    html_path = "reports/deepchecks_report.html"
    result.save_as_html(html_path)

    # 2. Плоский Markdown-отчёт без JS можно читать в VS Code
    md_path = "reports/deepchecks_report.md"
    result.save_as_cml_markdown(md_path)

    print(f"HTML-отчёт Deepchecks сохранён в {html_path}")
    print(f"Markdown-отчёт Deepchecks сохранён в {md_path}")


if __name__ == "__main__":
    main()