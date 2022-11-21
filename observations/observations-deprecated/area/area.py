import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('./../../csvs/Mumbai_Dec_17-23.csv', delimiter='	')
print("Area for Bhopal")
l = df.to_dict('records')


orders = []
for dic in l:
	orders.append(dic)

mini_lat = 4321.0
mini_long = 4321.0
max_lat = 0.0
max_long = 0.0

for order in orders:
    lat = order['DELIVERY_LAT']
    longi = order['DELIVERY_LNG']
    mini_lat = min(mini_lat, lat)
    mini_long = min(mini_long, longi)

    max_lat = max(max_lat, lat)
    max_long = max(max_long, longi)

print("Lowest Latitude: "+str(mini_lat))
print("Highest Latitude: "+str(max_lat))
print("Lowest Longitude: "+str(mini_long))
print("Highest Longitude: "+str(max_long))