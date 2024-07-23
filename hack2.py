# Import necessary libraries
import holidays
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# veri setini yüklediğim kısım
file_path = 'h.xlsx'
df = pd.read_excel(file_path)

# tatilleri belirttiğimiz kısım
turkey_holidays = holidays.Turkey(years=[2023, 2024])


df['TIME_STAMP'] = pd.to_datetime(df['TIME_STAMP'])

df['weekday'] = df['TIME_STAMP'].dt.weekday
df['hour'] = df['TIME_STAMP'].dt.hour
df['minute'] = df['TIME_STAMP'].dt.minute

# tatil günü kontrolü
df['is_holiday'] = df['TIME_STAMP'].dt.normalize().isin(turkey_holidays)
# haftasonu kontrolü
df['is_weekend'] = df['TIME_STAMP'].dt.weekday >= 5

# veri setinde gördüğümüz eksik kısımları burda sütunun ortalama değerlerini kullanarak dolduruyoruz
df['DOWNLOAD'].fillna(df['DOWNLOAD'].mean(), inplace=True)
df['UPLOAD'].fillna(df['UPLOAD'].mean(), inplace=True)

#Download için  ARIMA modelinin fit edilmesi 
model_download = ARIMA(df['DOWNLOAD'], order=(5, 1, 0))
model_download_fit = model_download.fit()

#Upload   ARIMA modelinin fit edilmesi
model_upload = ARIMA(df['UPLOAD'], order=(5, 1, 0))
model_upload_fit = model_upload.fit()

# gerçek ve tahmin edilen değerler arasındaki fark
df['forecast_download'] = model_download_fit.fittedvalues
df['forecast_upload'] = model_upload_fit.fittedvalues

df['residual_download'] = df['DOWNLOAD'] - df['forecast_download']
df['residual_upload'] = df['UPLOAD'] - df['forecast_upload']

# anomali eşiğği tanımlama
threshold_download = 3 * np.std(df['residual_download'])
threshold_upload = 3 * np.std(df['residual_upload'])
df['anomaly_download'] = np.abs(df['residual_download']) > threshold_download
df['anomaly_upload'] = np.abs(df['residual_upload']) > threshold_upload

# Download anomali grafiği
plt.figure(figsize=(14, 7))
plt.plot(df['TIME_STAMP'], df['DOWNLOAD'], label='Download')
plt.scatter(df[df['anomaly_download']]['TIME_STAMP'], df[df['anomaly_download']]['DOWNLOAD'], color='red', label='Anomalies')
plt.xlabel('Time')
plt.ylabel('Download Speed')
plt.title('Download Speed Anomalies')
plt.legend()
plt.show()

# Upload anomali grafiği
plt.figure(figsize=(14, 7))
plt.plot(df['TIME_STAMP'], df['UPLOAD'], label='Upload')
plt.scatter(df[df['anomaly_upload']]['TIME_STAMP'], df[df['anomaly_upload']]['UPLOAD'], color='red', label='Anomalies')
plt.xlabel('Time')
plt.ylabel('Upload Speed')
plt.title('Upload Speed Anomalies')
plt.legend()
plt.show()

# Anomali sayısı
num_anomalies_download = df['anomaly_download'].sum()
num_anomalies_upload = df['anomaly_upload'].sum()

print(f"Number of anomalies in Download Speed: {num_anomalies_download}")
print(f"Number of anomalies in Upload Speed: {num_anomalies_upload}")
