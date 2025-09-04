# Study-TaxiOrderForecasting

## �������

### ����� �������
������ ���������� ��� �������� �׸������� ����� � ����� ��������������� ���������� ������� ����� � ���������� �� ��������� ���. ��� �������� �������������� ����������� ��������� � ������� ������� ��������. ������������ ������������ ������ �� ����� `taxi.csv`, ������ �������� RMSE ? 48 �� �������� �������, ��� ������������� �����������.

### �������
- **����**: `taxi.csv`
- **�������**:
  - `datetime`: ��������� ����� ������� (���������� � ���������� 10 �����, �������������� �� ����).
  - `num_orders`: ���������� ������� �����.

### �����������
1. **���������� ������**:
   - ��������������� ������ �� ����.
   - ��������� �������� ������� ��������� �� IQR.
   - �������� ���������: ����������� (`hour`, `dayofweek`, `is_weekend`), ���� (1�24 ����), ���������� ���������� (mean/std ��� ���� 3, 6, 12, 24 ����).
2. **����������������� ������ ������ (EDA)**:
   - �������� �������� ����������, ������ ���������� ����� (�������� � �������), ������� (>200 �������).
   - ������������ 24-������� ������������� ����� ������������ � ��������������.
3. **�������� �������**:
   - ������������ `LinearRegression`, `LGBMRegressor`, `CatBoostRegressor`.
   - ������ ��������������� ����� `RandomizedSearchCV` � `TimeSeriesSplit`.
   - ������ ������: `CatBoostRegressor` (RMSE �� �����-���������: 24.17).
4. **������������**:
   - Train RMSE: 19.01
   - Test RMSE: 31.70 (������������� ���������� RMSE ? 48).

### ���������
1. ���������� �����������:
   ```bash
   git clone https://github.com/legonc/Study-TaxiOrderForecasting.git
   ```
2. ���������� ����������� �� `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

### �������������
1. ��������� `taxi.csv` � ����� ������� ��� �������� ���� � ��������.
2. ��������� Jupyter Notebook `taxi_forecasting.ipynb` � ��������� ��� ������.

### ��������� �������
- `taxi_forecasting.ipynb`: �������� ������� � ����� �������.
- `taxi.csv`: ������� � ������� � ������� �����.
- `requirements.txt`: ������ ������������ ��� Python.
- `.gitignore`: ��������� ��������� �����, ���� � �������.

### ����������
- **������ ������**: `CatBoostRegressor`
  - ���������: `iterations=140`, `depth=5`, `learning_rate=0.1`, `scaler=RobustScaler`, `imputer_strategy=mean`.
  - Train RMSE: 19.01
  - Test RMSE: 31.70
- **�����**: ������ ������������� ����������� (RMSE ? 48), �� ������������ ������� ��������� (�������������, ������ ������).

### ������������
- ������� ������������� � `CatBoost` (��������, ��������� `l2_leaf_reg`).
- �������� �������� ��� ���������� � ������� � ����������.
- ���������������� ������ ��������������� � ������� Optuna.
- ��������� �������� � ������������� ������ ����� ������������� � �������� ���������.

### ��������
MIT License

---

## English

### Project Overview
This project was developed for the "׸������� �����" company to predict the number of taxi orders in airports for the next hour. The goal is to optimize driver allocation during peak demand periods. Using historical data from `taxi.csv`, the model achieves an RMSE ? 48 on the test set, meeting the project requirements.

### Dataset
- **File**: `taxi.csv`
- **Columns**:
  - `datetime`: Timestamp of orders (originally 10-minute intervals, resampled to hourly).
  - `num_orders`: Number of taxi orders.

### Methodology
1. **Data Preparation**:
   - Resampled data to hourly intervals.
   - Handled outliers using IQR clipping.
   - Created features: calendar (`hour`, `dayofweek`, `is_weekend`), lags (1�24 hours), rolling statistics (mean/std for 3, 6, 12, 24-hour windows).
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