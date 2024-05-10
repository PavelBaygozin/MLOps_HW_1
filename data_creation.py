
import numpy as np
import pandas as pd
import os

def generate_data(num_samples, noise_level=0.0, anomaly=False):
    np.random.seed(42)
    temperatures = np.random.normal(loc=20, scale=5, size=num_samples)
    if anomaly:
        anomalies = np.random.choice(range(num_samples), size=int(0.05 * num_samples), replace=False)
        temperatures[anomalies] += np.random.normal(loc=20, scale=5, size=len(anomalies))
    noise = np.random.normal(loc=0, scale=noise_level, size=num_samples)
    days = np.arange(num_samples)
    return pd.DataFrame({'Day': days, 'Temperature': temperatures + noise})

def save_data(data, folder, filename):
    if not os.path.exists(folder):
        os.makedirs(folder)
    filepath = os.path.join(folder, filename)
    data.to_csv(filepath, index=False)

# Generate and save training data
train_data_normal = generate_data(1000)
train_data_noisy = generate_data(1000, noise_level=2)
train_data_anomaly = generate_data(1000, anomaly=True)
save_data(train_data_normal, 'train', 'train_normal.csv')
save_data(train_data_noisy, 'train', 'train_noisy.csv')
save_data(train_data_anomaly, 'train', 'train_anomaly.csv')

# Generate and save test data
test_data_normal = generate_data(300)
test_data_noisy = generate_data(300, noise_level=2)
test_data_anomaly = generate_data(300, anomaly=True)
save_data(test_data_normal, 'test', 'test_normal.csv')
save_data(test_data_noisy, 'test', 'test_noisy.csv')
save_data(test_data_anomaly, 'test', 'test_anomaly.csv')
