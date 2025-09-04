# Study-TaxiOrderForecasting

## Русский

### Обзор проекта
Проект разработан для компании «Чётенькое такси» с целью прогнозирования количества заказов такси в аэропортах на следующий час. Это помогает оптимизировать привлечение водителей в периоды пиковой нагрузки. Использованы исторические данные из файла `taxi.csv`, модель достигла RMSE ? 48 на тестовой выборке, что соответствует требованиям.

### Датасет
- **Файл**: `taxi.csv`
- **Столбцы**:
  - `datetime`: Временная метка заказов (изначально с интервалом 10 минут, ресемплировано по часу).
  - `num_orders`: Количество заказов такси.

### Методология
1. **Подготовка данных**:
   - Ресемплирование данных по часу.
   - Обработка выбросов методом клиппинга по IQR.
   - Создание признаков: календарные (`hour`, `dayofweek`, `is_weekend`), лаги (1–24 часа), скользящие статистики (mean/std для окон 3, 6, 12, 24 часа).
2. **Исследовательский анализ данных (EDA)**:
   - Выявлены суточная сезонность, слабый восходящий тренд (особенно в августе), выбросы (>200 заказов).
   - Подтверждена 24-часовая периодичность через декомпозицию и автокорреляцию.
3. **Обучение моделей**:
   - Использованы `LinearRegression`, `LGBMRegressor`, `CatBoostRegressor`.
   - Подбор гиперпараметров через `RandomizedSearchCV` с `TimeSeriesSplit`.
   - Лучшая модель: `CatBoostRegressor` (RMSE на кросс-валидации: 24.17).
4. **Тестирование**:
   - Train RMSE: 19.01
   - Test RMSE: 31.70 (удовлетворяет требованию RMSE ? 48).

### Установка
1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/legonc/Study-TaxiOrderForecasting.git
   ```
2. Установите зависимости из `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

### Использование
1. Поместите `taxi.csv` в папку проекта или обновите путь в ноутбуке.
2. Запустите Jupyter Notebook `taxi_forecasting.ipynb` и выполните все ячейки.

### Структура проекта
- `taxi_forecasting.ipynb`: Основной ноутбук с кодом проекта.
- `taxi.csv`: Датасет с данными о заказах такси.
- `requirements.txt`: Список зависимостей для Python.
- `.gitignore`: Исключает временные файлы, логи и датасет.

### Результаты
- **Лучшая модель**: `CatBoostRegressor`
  - Параметры: `iterations=140`, `depth=5`, `learning_rate=0.1`, `scaler=RobustScaler`, `imputer_strategy=mean`.
  - Train RMSE: 19.01
  - Test RMSE: 31.70
- **Вывод**: Модель удовлетворяет требованиям (RMSE ? 48), но переобучение требует доработки (регуляризация, анализ данных).

### Рекомендации
- Усилить регуляризацию в `CatBoost` (например, увеличить `l2_leaf_reg`).
- Добавить признаки для праздников и событий в аэропортах.
- Автоматизировать подбор гиперпараметров с помощью Optuna.
- Проверить различия в распределении данных между тренировочной и тестовой выборками.

### Лицензия
MIT License

---

## English

### Project Overview
This project was developed for the "Чётенькое такси" company to predict the number of taxi orders in airports for the next hour. The goal is to optimize driver allocation during peak demand periods. Using historical data from `taxi.csv`, the model achieves an RMSE ? 48 on the test set, meeting the project requirements.

### Dataset
- **File**: `taxi.csv`
- **Columns**:
  - `datetime`: Timestamp of orders (originally 10-minute intervals, resampled to hourly).
  - `num_orders`: Number of taxi orders.

### Methodology
1. **Data Preparation**:
   - Resampled data to hourly intervals.
   - Handled outliers using IQR clipping.
   - Created features: calendar (`hour`, `dayofweek`, `is_weekend`), lags (1–24 hours), rolling statistics (mean/std for 3, 6, 12, 24-hour windows).
2. **Exploratory Data Analysis (EDA)**:
   - Identified daily seasonality, a slight upward trend (especially in August), and outliers (>200 orders).
   - Confirmed 24-hour periodicity via decomposition and autocorrelation.
3. **Model Training**:
   - Trained `LinearRegression`, `LGBMRegressor`, and `CatBoostRegressor`.
   - Tuned hyperparameters using `RandomizedSearchCV` with `TimeSeriesSplit`.
   - Best model: `CatBoostRegressor` (cross-validation RMSE: 24.17).
4. **Testing**:
   - Train RMSE: 19.01
   - Test RMSE: 31.70 (meets RMSE ? 48 requirement).

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/legonc/Study-TaxiOrderForecasting.git
   ```
2. Install dependencies from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
1. Place `taxi.csv` in the project directory or update the path in the notebook.
2. Run the Jupyter Notebook `taxi_forecasting.ipynb` and execute all cells.

### Project Structure
- `taxi_forecasting.ipynb`: Main notebook with project code.
- `taxi.csv`: Dataset with taxi order data.
- `requirements.txt`: List of Python dependencies.
- `.gitignore`: Excludes temporary files, logs, and dataset.

### Results
- **Best Model**: `CatBoostRegressor`
  - Parameters: `iterations=140`, `depth=5`, `learning_rate=0.1`, `scaler=RobustScaler`, `imputer_strategy=mean`.
  - Train RMSE: 19.01
  - Test RMSE: 31.70
- **Conclusion**: The model meets the requirement (RMSE ? 48), but overfitting needs further improvement (regularization, data analysis).

### Recommendations
- Enhance regularization in `CatBoost` (e.g., increase `l2_leaf_reg`).
- Add features for holidays and airport events.
- Automate hyperparameter tuning with Optuna.
- Investigate differences in data distribution between training and test sets.

### License
MIT License