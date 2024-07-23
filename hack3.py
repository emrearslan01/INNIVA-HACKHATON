import pandas as pd
import matplotlib.pyplot as plt

# veri setini yüklediğim kısım
file_path = 'h.xlsx'
df = pd.read_excel(file_path)

# okunabilir tarih ayarı
df['TIME_STAMP'] = pd.to_datetime(df['TIME_STAMP'])

# veri setinde gördüğümüz eksik kısımları burda sütunun ortalama değerlerini kullanarak dolduruyoruz
df['DOWNLOAD'].fillna(df['DOWNLOAD'].mean(), inplace=True)
df['UPLOAD'].fillna(df['UPLOAD'].mean(), inplace=True)

# saati çıkarıyoruz
df['hour'] = df['TIME_STAMP'].dt.hour

# her saat için toplam trafiği belirtiyoruz
df['total_traffic'] = df['DOWNLOAD'] + df['UPLOAD']

# 1 saatlik aralıklarla gruplandırın
traffic_by_hour = df.groupby('hour')['total_traffic'].sum().reset_index()

# azalan sırada sıralıyoruz
traffic_by_hour_sorted = traffic_by_hour.sort_values(by='total_traffic', ascending=False)
print(traffic_by_hour_sorted)

# Trafiği sütun grafiği halinde sunuyoruz
plt.figure(figsize=(12, 6))
plt.bar(traffic_by_hour['hour'].astype(str), traffic_by_hour['total_traffic'])
plt.xlabel('Hour of the Day (1-hour intervals)')
plt.ylabel('Total Traffic')
plt.title('Total Traffic by 1-Hour Intervals')
plt.grid(True)
plt.show()

traffic_by_hour_sorted
