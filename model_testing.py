import pandas as pd
import joblib
import os

def test_model(data_folder, model_path):
    model = joblib.load(model_path)
    data_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    for file in data_files:
        data = pd.read_csv(os.path.join(data_folder, file))
        X = data.drop('Temperature', axis=1)
        y = data['Temperature']
        score = model.score(X, y)
        print(f"Model score for {file}: {score}")

test_model('test_scaled', 'linear_regression_model.pkl')