# 🌦️ Weather Classification ML Project
Machine learning project for binary classification of weather conditions (precipitation/no precipitation) based on historical meteorological data. Solves the problem of precipitation forecasting using various classification algorithms.

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.0-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-red)
![Accuracy](https://img.shields.io/badge/Accuracy-93.4%25-brightgreen)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.956-success)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## Quick Start
To explore the full analysis interactively, open the notebook in your browser:
[![Open in NBViewer](https://img.shields.io/badge/Open%20in-NBViewer-orange?logo=jupyter&logoColor=orange)](https://nbviewer.org/github/VeronikaKolimova/weather-classification-ml-project/blob/main/Lab02_release2.ipynb
)

##  📌 Описание проекта
Проект машинного обучения для бинарной классификации погодных условий (осадки/без осадков) на основе исторических метеорологических данных. Решает задачу прогнозирования осадков с использованием различных алгоритмов классификации.

##  Цель проекта
Построить модель машинного обучения, способную предсказывать наличие осадков по метеорологическим параметрам с максимальной точностью.

##  Ключевые задачи
- Предобработка и анализ метеорологических данных
- Исследование баланса классов (дождливые/бездождливые дни)
- Сравнение различных алгоритмов классификации
- Обработка дисбаланса классов с помощью SMOTE
- Оценка моделей с использованием метрик качества

##  Данные
- Источник: исторические данные погоды
- Целевая переменная: бинарный признак осадков (0 - нет осадков, 1 - есть осадки)
- Признаки: температура, влажность, давление, скорость ветра, точка росы, видимость
- Размер: 8784 наблюдения

##  Используемые технологии
- **Python** 3.8+
- **Scikit-learn** — ML алгоритмы (RandomForest, LogisticRegression, SVM, XGBoost)
- **Pandas/Numpy** — обработка данных
- **Matplotlib/Seaborn** — визуализация
- **Imbalanced-learn** — обработка дисбаланса классов
- **XGBoost** — градиентный бустинг



