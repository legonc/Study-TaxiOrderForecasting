# Study-TaxiOrderForecasting

## Русский

### Обзор проекта
Проект разработан для компании «Чётенькое такси» с целью прогнозирования количества заказов такси в аэропортах на следующий час, чтобы оптимизировать привлечение водителей в периоды пиковой нагрузки. Использованы исторические данные из файла `taxi.csv` за период с марта по август 2018 года. Основная задача — построить модель с метрикой RMSE на тестовой выборке не более 48. Лучшая модель (`CatBoostRegressor`) достигла RMSE 42.58, что соответствует требованиям.

### Датасет
- **Файл**: `taxi.csv`
- **Столбцы**:
  - `datetime`: Временная метка заказов (изначально с интервалом 10 минут, ресемплировано по часу).
  - `num_orders`: Количество заказов такси.
- **Размер**: 26,496 строк (до ресемплирования), 4,416 строк (после ресемплирования по часу).
- **Временной диапазон**: 2018-03-01 00:00:00 — 2018-08-31 23:00:00.
- **Статистика**: Среднее количество заказов — 84.42, медиана — 78, максимум — 462 (выбросы >200 заказов).

### Методология
1. **Подготовка данных**:
   - Данные ресемплированы по часу с агрегацией по сумме (`num_orders`).
   - Выбросы (>200 заказов) оставлены без клиппинга, так как они отражают реальные пики спроса (подтверждено EDA).
   - Созданы признаки в функции `create_features`:
     - **Календарные**: `hour`, `dayofweek`, `is_weekend` (учитывают суточную и недельную сезонность).
     - **Лаги**: 1–24 часа (`lag_1`–`lag_24`) для захвата исторических паттернов.
     - **Скользящие статистики**: `rolling_mean` и `rolling_std` для окон 3, 6, 12, 24 часа с `shift(1)` для исключения утечки данных.
   - Итоговый датафрейм (`df_new`): 4,368 строк после удаления NaN.

2. **Исследовательский анализ данных (EDA)**:
   - **Структура**: Пропусков и дубликатов нет, индекс с частотой 1 час.
   - **Визуализации**:
     - График за 6 месяцев: слабый восходящий тренд, особенно в августе (возвращение из отпусков).
     - График за неделю: суточная сезонность (пики ~100–150 заказов каждые 24 часа, спад ночью/утром).
     - Гистограмма и боксплот: левое смещение, медиана ~100, IQR 54–107, выбросы >200 (121 случай, 2.74%).
   - **Декомпозиция**: Подтверждена суточная сезонность (период 24 часа) с возрастающей амплитудой. Остатки случайны, пики около нуля.
   - **Автокорреляция (ACF)**: Значимые пики на лагах, кратных 24, подтверждают сезонность.

3. **Корреляционный анализ и создание признаков**:
   - **Корреляция**: Высокая корреляция (|ρ| > 0.9) между `rolling_mean_12` и `rolling_mean_24`. Удаление признаков на основе корреляции не проводилось, так как все признаки информативны.
   - **Исправления**: Устранена утечка данных (добавлен `shift(1)` для rolling-статистик). Признаки создаются в функции `create_features`, возвращающей новый датафрейм.

4. **Обучение моделей**:
   - **Разделение данных**: Тестовая выборка — 10% (437 строк), без перемешивания для сохранения временной структуры.
   - **Baseline (LinearRegression)**:
     - Train RMSE: 25.86
   - **Подбор гиперпараметров**:
     - Использован `RandomizedSearchCV` с `TimeSeriesSplit` (5 фолдов) для трёх моделей: `LinearRegression`, `LGBMRegressor`, `CatBoostRegressor`.
     - **LinearRegression**:
       - RMSE на кросс-валидации: 27.23
       - Параметры: `fit_intercept=True`, `imputer_strategy=mean`, `scaler=RobustScaler`
       - Время: ~15.1 сек
     - **LGBMRegressor**:
       - RMSE на кросс-валидации: 25.7
       - Параметры: `n_estimators=70`, `max_depth=-1`, `learning_rate=0.1`, `imputer_strategy=mean`, `scaler=RobustScaler`
       - Время: ~3 мин 2 сек
     - **CatBoostRegressor**:
       - RMSE на кросс-валидации: 25.38
       - Параметры: `iterations=140`, `depth=5`, `learning_rate=0.1`, `imputer_strategy=mean`, `scaler=RobustScaler`
       - Время: ~1 мин 12 сек
   - **Вывод**: `CatBoostRegressor` показала лучший результат на кросс-валидации (RMSE 25.38).

5. **Тестирование**:
   - **Лучшая модель**: `CatBoostRegressor` (параметры: `iterations=140`, `depth=5`, `learning_rate=0.1`, `imputer_strategy=mean`, `scaler=RobustScaler`).
   - **Результаты**:
     - Train RMSE: 17.1
     - Test RMSE: 42.58
   - **Визуализация**:
     - Графики исходного и предсказанного рядов на тестовой выборке (весь период и первая неделя) показывают, что модель улавливает суточную сезонность, но ошибается на пиках спроса.
   - **Анализ**:
     - Test RMSE (42.58) удовлетворяет требованию (<48).
     - Разница между Train и Test RMSE (17.1 vs 42.58) указывает на переобучение.
     - Возможные причины: различия в распределении данных или недостаточная регуляризация.

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
  - Параметры: `iterations=140`, `depth=5`, `learning_rate=0.1`, `imputer_strategy=mean`, `scaler=RobustScaler`.
  - Train RMSE: 17.1
  - Test RMSE: 42.58
- **Вывод**: Модель удовлетворяет требованиям (RMSE < 48), но переобучение требует доработки через регуляризацию и анализ данных.

### Рекомендации
- **Улучшение модели**:
  - Усилить регуляризацию в `CatBoost` (например, увеличить `l2_leaf_reg`).
  - Попробовать ансамблевые методы (комбинация `CatBoost` и `LGBM`).
- **Анализ данных**:
  - Проверить различия в распределении тренировочных и тестовых данных.
  - Исследовать влияние выбросов (>200 заказов) на тестовую выборку.
- **Feature engineering**:
  - Добавить признаки для праздников и событий в аэропортах.
  - Рассмотреть нелинейные комбинации признаков или лаги с периодами >24 часов (например, 48 часов).
- **Масштабирование**:
  - Оптимизировать предобработку (например, кэширование `SimpleImputer`).
  - Автоматизировать подбор гиперпараметров с помощью `Optuna` для больших датасетов.

### Лицензия
MIT License

---

## English

### Project Overview
This project was developed for the "Чётенькое такси" company to predict the number of taxi orders in airports for the next hour, optimizing driver allocation during peak demand periods. Using historical data from `taxi.csv` (March to August 2018), the best model (`CatBoostRegressor`) achieved an RMSE of 42.58 on the test set, meeting the requirement of RMSE < 48.

### Dataset
- **File**: `taxi.csv`
- **Columns**:
  - `datetime`: Timestamp of orders (originally 10-minute intervals, resampled to hourly).
  - `num_orders`: Number of taxi orders.
- **Size**: 26,496 rows (before resampling), 4,416 rows (after hourly resampling).
- **Time Range**: 2018-03-01 00:00:00 — 2018-08-31 23:00:00.
- **Statistics**: Mean orders — 84.42, median — 78, maximum — 462 (outliers >200 orders).

### Methodology
1. **Data Preparation**:
   - Data resampled to hourly intervals with sum aggregation (`num_orders`).
   - Outliers (>200 orders) retained, as they reflect real demand peaks (confirmed by EDA).
   - Features created in the `create_features` function:
     - **Calendar**: `hour`, `dayofweek`, `is_weekend` (capturing daily/weekly seasonality).
     - **Lags**: 1–24 hours (`lag_1`–`lag_24`) for historical patterns.
     - **Rolling Statistics**: `rolling_mean` and `rolling_std` for 3, 6, 12, 24-hour windows with `shift(1)` to prevent data leakage.
   - Final dataset (`df_new`): 4,368 rows after dropping NaN.

2. **Exploratory Data Analysis (EDA)**:
   - **Structure**: No missing values or duplicates, index confirmed with 1-hour frequency.
   - **Visualizations**:
     - 6-month plot: Slight upward trend, especially in August (likely due to vacation returns).
     - Weekly plot: Daily seasonality (peaks ~100–150 orders every 24 hours, dips at night/morning).
     - Histogram and boxplot: Left-skewed distribution, median ~100, IQR 54–107, outliers >200 (121 cases, 2.74%).
   - **Decomposition**: Confirmed 24-hour seasonality with increasing amplitude. Residuals are random, peaking near zero.
   - **Autocorrelation (ACF)**: Significant peaks at lags divisible by 24, confirming seasonality.

3. **Correlation Analysis and Feature Engineering**:
   - **Correlation**: High correlation (|ρ| > 0.9) between `rolling_mean_12` and `rolling_mean_24`. No features were dropped based on correlation to retain informativeness.
   - **Fixes**: Eliminated data leakage by adding `shift(1)` to rolling statistics. Features created in `create_features`, returning a new dataframe.

4. **Model Training**:
   - **Data Split**: Test set — 10% (437 rows), no shuffling to preserve time structure.
   - **Baseline (LinearRegression)**:
     - Train RMSE: 25.86
   - **Hyperparameter Tuning**:
     - Used `RandomizedSearchCV` with `TimeSeriesSplit` (5 folds) for three models: `LinearRegression`, `LGBMRegressor`, `CatBoostRegressor`.
     - **LinearRegression**:
       - Cross-validation RMSE: 27.23
       - Parameters: `fit_intercept=True`, `imputer_strategy=mean`, `scaler=RobustScaler`
       - Time: ~15.1 sec
     - **LGBMRegressor**:
       - Cross-validation RMSE: 25.7
       - Parameters: `n_estimators=70`, `max_depth=-1`, `learning_rate=0.1`, `imputer_strategy=mean`, `scaler=RobustScaler`
       - Time: ~3 min 2 sec
     - **CatBoostRegressor**:
       - Cross-validation RMSE: 25.38
       - Parameters: `iterations=140`, `depth=5`, `learning_rate=0.1`, `imputer_strategy=mean`, `scaler=RobustScaler`
       - Time: ~1 min 12 sec
   - **Conclusion**: `CatBoostRegressor` showed the best cross-validation RMSE (25.38).

5. **Testing**:
   - **Best Model**: `CatBoostRegressor` (parameters: `iterations=140`, `depth=5`, `learning_rate=0.1`, `imputer_strategy=mean`, `scaler=RobustScaler`).
   - **Results**:
     - Train RMSE: 17.1
     - Test RMSE: 42.58
   - **Visualization**:
     - Plots of actual vs. predicted time series on the test set (full period and first week) show that the model captures daily seasonality but struggles with demand peaks.
   - **Analysis**:
     - Test RMSE (42.58) meets the requirement (<48).
     - Gap between Train and Test RMSE (17.1 vs 42.58) indicates overfitting.
     - Possible causes: Data distribution differences or insufficient regularization.

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
  - Parameters: `iterations=140`, `depth=5`, `learning_rate=0.1`, `imputer_strategy=mean`, `scaler=RobustScaler`.
  - Train RMSE: 17.1
  - Test RMSE: 42.58
- **Conclusion**: The model meets the requirement (RMSE < 48), but overfitting needs further improvement through regularization and data analysis.

### Recommendations
- **Model Improvement**:
  - Enhance regularization in `CatBoost` (e.g., increase `l2_leaf_reg`).
  - Experiment with ensemble methods (e.g., combining `CatBoost` and `LGBM`).
- **Data Analysis**:
  - Investigate differences in training and test data distributions.
  - Analyze the impact of outliers (>200 orders) on the test set.
- **Feature Engineering**:
  - Add features for holidays and airport events.
  - Consider nonlinear feature combinations or longer lag periods (e.g., 48 hours).
- **Scalability**:
  - Optimize preprocessing (e.g., cache `SimpleImputer`).
  - Automate hyperparameter tuning with `Optuna` for larger datasets.

### License
MIT License