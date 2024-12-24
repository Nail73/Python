import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Словарь для преобразования строковых месяцев в числовые
month_mapping = {
    'Январь': 1, 'Февраль': 2, 'Март': 3, 'Апрель': 4,
    'Май': 5, 'Июнь': 6, 'Июль': 7, 'Август': 8,
    'Сентябрь': 9, 'Октябрь': 10, 'Ноябрь': 11, 'Декабрь': 12
}

# Загрузка данных из Excel-файла
def load_data(file_path):
    try:
        data = pd.read_excel(file_path)
        return data
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return None

# Предобработка данных
def preprocess_data(data):
    try:
        # Убедимся, что данные содержат нужные столбцы
        if 'Shop' not in data.columns or 'Year' not in data.columns or 'Month' not in data.columns or 'Sales' not in data.columns:
            raise ValueError("Отсутствуют необходимые столбцы в данных")

        # Преобразование строковых месяцев в числовые
        data['Month'] = data['Month'].map(month_mapping)

        # Добавление дополнительных признаков
        data['Year_Month'] = data['Year'] * 12 + data['Month']
        data['Month_Sin'] = np.sin(2 * np.pi * data['Month'] / 12)
        data['Month_Cos'] = np.cos(2 * np.pi * data['Month'] / 12)

        return data
    except Exception as e:
        print(f"Ошибка при предобработке данных: {e}")
        return None

# Обучение и оценка модели для каждого магазина с использованием Grid Search
def train_and_evaluate_models(data):
    models = {}
    evaluation = {}
    try:
        shops = data['Shop'].unique()
        for shop in shops:
            shop_data = data[data['Shop'] == shop]
            X = shop_data[['Year', 'Month', 'Year_Month', 'Month_Sin', 'Month_Cos']]
            y = shop_data['Sales']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Настройка гиперпараметров с использованием Grid Search
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 4, 5],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            gb = GradientBoostingRegressor(random_state=42)
            grid_search = GridSearchCV(estimator=gb, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_

            y_pred = best_model.predict(X_test)
            models[shop] = best_model
            evaluation[shop] = {
                'MSE': mean_squared_error(y_test, y_pred),
                'MAE': mean_absolute_error(y_test, y_pred),
                'R2': r2_score(y_test, y_pred)
            }
        return models, evaluation
    except Exception as e:
        print(f"Ошибка при обучении и оценке модели: {e}")
        return None, None

# Прогнозирование для каждого магазина
def predict_future_sales(models, start_year, end_year):
    future_dates = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            year_month = year * 12 + month
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)
            future_dates.append([year, month, year_month, month_sin, month_cos])
    future_dates = np.array(future_dates)

    predictions = {}
    for shop, model in models.items():
        predictions[shop] = model.predict(future_dates)
    return future_dates, predictions

# Основная функция
def main(file_path, start_year, end_year):
    data = load_data(file_path)
    if data is None:
        return

    data = preprocess_data(data)
    if data is None:
        return

    models, evaluation = train_and_evaluate_models(data)
    if models is None:
        return

    # Вывод оценки моделей
    for shop, metrics in evaluation.items():
        print(f"Shop: {shop}")
        print(f"MSE: {metrics['MSE']}")
        print(f"MAE: {metrics['MAE']}")
        print(f"R2: {metrics['R2']}")
        print()

    future_dates, predictions = predict_future_sales(models, start_year, end_year)

    # Создание DataFrame для прогнозов
    all_predictions = []
    for shop, pred in predictions.items():
        for i, date in enumerate(future_dates):
            all_predictions.append([shop, date[0], date[1], pred[i]])

    future_sales = pd.DataFrame(all_predictions, columns=['Shop', 'Year', 'Month', 'Predicted_Sales'])

    print(future_sales)

    # Сохранение прогнозов в новый Excel-файл
    output_file_path = 'future_sales_predictions.xlsx'
    future_sales.to_excel(output_file_path, index=False)
    print(f"Прогнозы сохранены в файл: {output_file_path}")

# Пример использования
if __name__ == "__main__":
    file_path = 'sales_data.xlsx'  # Путь к вашему Excel-файлу
    start_year = 2025  # Начальный год для прогноза
    end_year = 2025  # Конечный год для прогноза
    main(file_path, start_year, end_year)
