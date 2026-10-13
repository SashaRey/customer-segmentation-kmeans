# customer-segmentation-kmeans

Kaggle-ориентированный шаблон ML-проекта для задачи сегментации клиентов с использованием K-Means.

## Quick Start

1. Установите зависимости:

```bash
pip install -r requirements.txt
```

2. Откройте основной ноутбук:

```bash
jupyter notebook notebooks/solution.ipynb
```

3. Выполните шаги по порядку: загрузка данных, EDA, подготовка признаков, обучение, оценка и генерация submission.

## Project Structure

```text
customer-segmentation-kmeans/
├── data/                  # Данные Kaggle (сырые и/или подготовленные)
├── notebooks/
│   ├── solution.ipynb     # Единый основной ноутбук с полным workflow
│   └── analysis.ipynb     # Дополнительные черновые исследования (опционально)
├── results/               # Артефакты: предсказания, графики, таблицы метрик
├── src/
│   ├── __init__.py
│   └── utils.py           # Заготовка под будущий рефакторинг кода из ноутбука
├── requirements.txt
├── .gitignore
└── README.md
```

Почему код в одном ноутбуке:
- Для Kaggle это ускоряет итерации и упрощает воспроизводимость эксперимента.
- Весь pipeline виден в одном месте: от загрузки данных до submission.
- При росте проекта логику можно постепенно переносить в `src/`, не ломая основной workflow.

## Dependencies

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
