from pathlib import Path

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently import ColumnMapping

from data import load_and_split_data


def main():
    # Подгружаем те же данные, что и для модели
    train_df, test_df = load_and_split_data()

    # reference = train, current = test
    reference_data = train_df.copy()
    current_data = test_df.copy()

    # Явно описываем маппинг колонок для Evidently
    feature_cols = [c for c in reference_data.columns if c != "target"]
    column_mapping = ColumnMapping(
        target="target",          # целевая переменная
        prediction=None,          # пока без предсказаний модели
        numerical_features=feature_cols,
        categorical_features=[],  # в breast_cancer все признаки числовые
    )

    # Репорт по дрейфу данных
    report = Report(
        metrics=[
            DataDriftPreset()
        ]
    )

    report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )

    Path("reports").mkdir(exist_ok=True)
    html_path = "reports/evidently_drift_report.html"
    report.save_html(html_path)

    print(f"Отчёт Evidently сохранён в {html_path}")


if __name__ == "__main__":
    main()