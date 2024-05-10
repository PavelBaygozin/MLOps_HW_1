
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def preprocess_data(input_folder, output_folder):
    scaler = StandardScaler()
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(input_folder):
        data_path = os.path.join(input_folder, filename)
        data = pd.read_csv(data_path)
        scaled_data = scaler.fit_transform(data)
        output_path = os.path.join(output_folder, filename)
        pd.DataFrame(scaled_data, columns=data.columns).to_csv(output_path, index=False)

preprocess_data('train', 'train_scaled')
preprocess_data('test', 'test_scaled')
