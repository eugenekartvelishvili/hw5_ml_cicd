# HW5: ML pipeline с CI/CD

В работе реализован простой ML-пайплайн на датасете breast cancer из `sklearn` и настроен CI/CD в GitHub Actions и GitLab CI.

## Структура проекта

- `src/data.py` — загрузка датасета Breast Cancer, разбиение на train/test.
- `src/train.py` — обучение `RandomForestClassifier`, логирование метрик в MLflow, сохранение использованных данных.
- `src/checks.py` — запуск `deepchecks` Full Suite для train/test и сохранение отчёта.
- `src/drift.py` — отчёт по data drift с помощью `evidently` (DataDriftPreset).
- `requirements.txt` — все зависимости проекта.
- `Dockerfile` — образ для запуска обучения внутри контейнера.
- `.github/workflows/ci.yml` — GitHub Actions: `train.py` + `checks.py` + `drift.py`.
- `.gitlab-ci.yml` — GitLab CI: те же три шага внутри `python:3.11-slim`.

## Как запустить локально

```bash
python3 -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Обучение модели + MLflow логирование
python src/train.py

# Deepchecks-отчёт
python src/checks.py   # сохранит reports/deepchecks_report.html

# Отчёт по дрейфу данных
python src/drift.py    # сохранит reports/evidently_drift_report.html
