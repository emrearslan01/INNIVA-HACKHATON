import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
import holidays

# Veri setini yükleme
file_path = 'h.xlsx'
df = pd.read_excel(file_path)

# Tatilleri belirleme
turkey_holidays = holidays.Turkey(years=[2023, 2024])

# Tarih bilgisini okunabilir hale getirme
df['TIME_STAMP'] = pd.to_datetime(df['TIME_STAMP'])

# Zaman bilgilerini çıkartma
df['weekday'] = df['TIME_STAMP'].dt.weekday
df['hour'] = df['TIME_STAMP'].dt.hour
df['minute'] = df['TIME_STAMP'].dt.minute

# Tatil günü kontrolü
df['is_holiday'] = df['TIME_STAMP'].dt.normalize().isin(turkey_holidays)

# Hafta sonu kontrolü
df['is_weekend'] = df['TIME_STAMP'].dt.weekday >= 5

# Eksik değerleri doldurma (ortalamalarla)
df['DOWNLOAD'].fillna(df['DOWNLOAD'].mean(), inplace=True)
df['UPLOAD'].fillna(df['UPLOAD'].mean(), inplace=True)

# Aynı gün ve saat dilimindeki ortalama trafik yoğunluklarını hesaplama
day_time_grouped = df.groupby(['weekday', 'hour', 'minute']).agg({'DOWNLOAD': 'mean', 'UPLOAD': 'mean'}).reset_index()

# Tüm günler için ortalama değerlere göre anomali tespiti
df = pd.merge(df, day_time_grouped, on=['weekday', 'hour', 'minute'], suffixes=('', '_avg'), how='left')

# Fark hesaplama
df['download_diff'] = df['DOWNLOAD'] - df['DOWNLOAD_avg']
df['upload_diff'] = df['UPLOAD'] - df['UPLOAD_avg']

# Isolation Forest algoritması için özellikler
features = ['download_diff', 'upload_diff']

# Özelliklerin standartlaştırılması
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features].fillna(0))  # NaN değerlerini 0 ile dolduruyoruz

# Isolation Forest algoritmasını uyguluyoruz
isolation_forest = IsolationForest(contamination=0.08, random_state=42)
df['anomaly_score'] = isolation_forest.fit_predict(df_scaled)
df['anomaly'] = df['anomaly_score'] == -1

# Anomali ve normal verileri ayırma
df_anomalies = df[df['anomaly']]
df_non_anomalies = df[~df['anomaly']]

# Anomalilerin görselleştirilmesi (Download)
plt.figure(figsize=(14, 7))
plt.plot(df['TIME_STAMP'], df['DOWNLOAD'], label='Download', color='blue')
plt.scatter(df_anomalies['TIME_STAMP'], df_anomalies['DOWNLOAD'], color='red', label='Anomalies')
plt.xlabel('Time')
plt.ylabel('Download Speed')
plt.title('Anomalies in Download Speed Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Anomalilerin görselleştirilmesi (Upload)
plt.figure(figsize=(14, 7))
plt.plot(df['TIME_STAMP'], df['UPLOAD'], label='Upload', color='green')
plt.scatter(df_anomalies['TIME_STAMP'], df_anomalies['UPLOAD'], color='red', label='Anomalies')
plt.xlabel('Time')
plt.ylabel('Upload Speed')
plt.title('Anomalies in Upload Speed Over Time')
plt.legend()
plt.grid(True)
plt.show()

num_anomalies = df[df['anomaly'] == True].shape[0]
print(f"Number of anomalies: {num_anomalies}")

# Download ve uploadların ortalamalarını saatlik dakikalık olarak hafta içi, hafta sonu ve tatil günleri için hesaplayıp grafik haline getirme
traffic_grouped = df.groupby(['hour', 'minute', 'is_weekend', 'is_holiday']).agg({'DOWNLOAD': 'mean', 'UPLOAD': 'mean'}).reset_index()

# Weekday Traffic
plt.figure(figsize=(12, 6))
weekday_traffic = traffic_grouped[(traffic_grouped['is_weekend'] == False) & (traffic_grouped['is_holiday'] == False)]
plt.bar(weekday_traffic['hour'] - 0.2, weekday_traffic['DOWNLOAD'], width=0.4, label='Weekday Download', color='red')
plt.bar(weekday_traffic['hour'] + 0.2, weekday_traffic['UPLOAD'], width=0.4, label='Weekday Upload', color='blue')
plt.xlabel('Hour of the Day')
plt.ylabel('Average Traffic')
plt.xticks(range(0, 24, 1)) 
plt.title('Weekday Traffic by Hour')
plt.legend()
plt.grid(True)
plt.show()

# Weekend Traffic
plt.figure(figsize=(12, 6))
weekend_traffic = traffic_grouped[traffic_grouped['is_weekend'] == True]
plt.bar(weekend_traffic['hour'] - 0.2, weekend_traffic['DOWNLOAD'], width=0.4, label='Weekend Download', color='red')
plt.bar(weekend_traffic['hour'] + 0.2, weekend_traffic['UPLOAD'], width=0.4, label='Weekend Upload', color='blue')
plt.xlabel('Hour of the Day')
plt.ylabel('Average Traffic')
plt.xticks(range(0, 24, 1))  
plt.title('Weekend Traffic by Hour')
plt.legend()
plt.grid(True)
plt.show()

# Holiday Traffic
plt.figure(figsize=(12, 6))
holiday_traffic = traffic_grouped[traffic_grouped['is_holiday'] == True]
plt.bar(holiday_traffic['hour'] - 0.2, holiday_traffic['DOWNLOAD'], width=0.4, label='Holiday Download', color='red')
plt.bar(holiday_traffic['hour'] + 0.2, holiday_traffic['UPLOAD'], width=0.4, label='Holiday Upload', color='blue')
plt.xlabel('Hour of the Day')
plt.ylabel('Average Traffic')
plt.xticks(range(0, 24, 1))  
plt.title('Holiday Traffic by Hour')
plt.legend()
plt.grid(True)
plt.show()
