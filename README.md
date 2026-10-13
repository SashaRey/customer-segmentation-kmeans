# customer-segmentation-kmeans

Шаблон репозитория для ML-задачи на Kaggle с одним основным ноутбуком, в котором выполняется полный пайплайн: от загрузки данных до генерации submission.

## Quick Start

1. Установите зависимости:

```bash
pip install -r requirements.txt
```

2. Откройте основной ноутбук:

```bash
jupyter notebook notebooks/solution.ipynb
```

3. Выполняйте секции последовательно и сохраняйте артефакты в `results/`.

## Project Structure

```text
customer-segmentation-kmeans/
├── data/                   # Данные (локальные копии Kaggle-датасетов)
├── notebooks/
│   ├── solution.ipynb      # Основной ноутбук с полным решением
│   └── analysis.ipynb      # Дополнительные/черновые заметки (если нужны)
├── results/                # Предсказания, метрики, графики, submission-файлы
├── src/
│   ├── __init__.py
│   └── utils.py            # Будущие функции для рефакторинга из ноутбука
├── requirements.txt
├── .gitignore
└── README.md
```

Почему один основной ноутбук:
- Формат Kaggle обычно предполагает единый воспроизводимый notebook для проверки решения.
- Это упрощает демонстрацию полного процесса (EDA → feature engineering → training → evaluation → submission) в одном месте.
- По мере роста проекта повторяющийся код можно постепенно переносить в `src/` без потери удобства для соревнований.

## Main Notebook Sections

В `notebooks/solution.ipynb` подготовлены секции:
- Imports & Config
- Data Loading
- EDA (Exploratory Data Analysis)
- Feature Engineering
- Model Training
- Evaluation
- Submission Generation
