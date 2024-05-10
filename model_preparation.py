
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

def train_model(data_folder, model_path):
    model = LinearRegression()
    data_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    for file in data_files:
        data = pd.read_csv(os.path.join(data_folder, file))
        if 'Temperature' in data.columns:
            X = data.drop('Temperature', axis=1)  # Убедитесь, что есть другие столбцы кроме 'Temperature'
            y = data['Temperature']
            if X.empty:
                print(f"No features to train on in {file}.")
            else:
                model.fit(X, y)
                joblib.dump(model, model_path)
        else:
            print(f"'Temperature' column not found in {file}.")

train_model('train_scaled', 'linear_regression_model.pkl')
