import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
import sys

city = sys.argv[1]
total_slots = 24

def getTimeIndex(dt):
	current = datetime.datetime(2011, 1, 1)
	t1 = current.time()
	t2 = dt.time()
	for i in range(0, total_slots):
		current = current + datetime.timedelta(minutes = 60)
		if(current.time() > t2):
			return i
	return total_slots-1

def get_preparation_time(order):
	dt1 = datetime.datetime.strptime(order['ORDER_MAKELINE_DATE_TIME'], "%Y-%m-%d %H:%M:%S")
	dt2 = datetime.datetime.strptime(order['KITCHEN_DISPLAY_TIME'], "%Y-%m-%d %H:%M:%S")
	diff = dt1 - dt2
	return diff.total_seconds()/60

df = pd.read_excel(f'./../../LevelOneCleaning/{city}.xlsx')
l = df.to_dict('records')


allY = []
meanX = []
stdX = []

orders = []
for dic in l:
	# if(dic['ORDER_DATE'] == '17/12/2021'):
	orders.append(dic)

finalDict = []
for i in range(0, total_slots):
	finalDict.append([])

for order in orders:
	dt = datetime.datetime.strptime(order['ORDER_RECEIVED_DATE_TIME'], "%Y-%m-%d %H:%M:%S")
	i = getTimeIndex(dt)
	t = get_preparation_time(order)
	if (t > 51.0): # removing orders for which preparation time is > 1hr
		continue
	finalDict[i].append(t+9.0) #adding baking time (check doc)


X = [i for i in range(0, total_slots)]
y = []
e = []
for i in range(0, total_slots):
	if(len(finalDict[i]) == 0):
		y.append(0)
		e.append(0)
	else:
		y.append(np.mean(finalDict[i]))
		e.append(np.std(finalDict[i]))

# plt.plot(X, y)
plt.errorbar(X, y, e, fmt='-o')
plt.xticks(X)
plt.title(f"Avg food preperation time in each timeslot for 17-31 dec 2021 in {city}")
plt.xlabel('Time (in 24 hour format)')
plt.ylabel('Time in minutes')
plt.ylim(0, None)
# plt.savefig('individual_restaurants.png')
plt.savefig(f'./prep_time_{city}_minutes.png')

# points = {}

# # dictionary with 
# # key = restuarant_id, 
# # value = list of dictionaries of all orders
# res = {}



# # global X and Y (x,(y,number of occurences))
# g_X=[]
# g_Y=[]
# datapoints_g={}
# for i in range(0, len(l)):
# 	dic = l[i]
# 	if(dic['completion_status'] != 'Completed'):
# 		continue
# 	# if(dic['order_received_date_time'])
# 	# if(dic['food_prepared_time'] == '' or type(dic['food_prepared_time']) == float):
# 	# 	continue
# 	# if(dic['confirmed_time'] == '' or type(dic['confirmed_time']) == float):
# 	# 	continue
# 	# if(datetime.datetime.fromisoformat(dic['confirmed_time']) >= datetime.datetime.fromisoformat(dic['food_prepared_time'])):
# 	# 	continue
# 	rid = dic['store_id']
# 	if((rid, dic['order_date']) in res):
# 		res[(rid, dic['order_date'])].append(dic)
# 	else:
# 		res[(rid, dic['order_date'])] = [dic]

# #dictionary : key = restaurant_id and value = list of load
# final = {}

# for (key, date) in list(res.keys()):
# 	# print(len(res[key]))
# 	# print(res[key])
# 	# preparation_time = [] #48 size
# 	NoOfItems = [] #48 size
# 	for i in range(0, 48):
# 		# preparation_time.append(0)
# 		NoOfItems.append(0)
# 	ls = res[(key, date)]
# 	for dic in ls:
# 		# dt = Convert.ToDateTime((dic['confirmed_time'])
# 		ct = datetime.datetime.fromisoformat(dic['order_received_date_time'])
# 		# fpt = datetime.datetime.fromisoformat(dic['food_prepared_time'])
# 		# nt = ((fpt-ct).total_seconds())/60
# 		print(ct)
# 		TimeIndex = getTimeIndex(ct)
# 		# print("TimeIndex is : ")
# 		# print(TimeIndex)
# 		# preparation_time[TimeIndex] += dic['item_count']*nt
# 		NoOfItems[TimeIndex] += 1 #dic['item_count']
	
# 	# datapoints={}# (x,(y,number of occurences))
# 	# for i in range(0, 48):
# 	# 	if(NoOfItems[i] != 0):
# 	# 		# preparation_time[i] = preparation_time[i]/NoOfItems[i]
# 	# 		if NoOfItems[i] in datapoints:
# 	# 			datapoints[NoOfItems[i]] = (datapoints[NoOfItems[i]][0]+preparation_time[i],datapoints[NoOfItems[i]][1]+1)
# 	# 			# datapoints[NoOfItems[i]][0]+=NoOfItems
# 	# 			# datapoints[NoOfItems[i]][1]+=1
# 	# 		else:
# 	# 			datapoints[NoOfItems[i]]=(preparation_time[i],1)
# 	# 	else:
# 	# 		preparation_time[i] = 0
# 	# X=[]
# 	# Y=[]
# 	# for x in datapoints.keys():
# 	# 	X.append(x)
# 	# 	Y.append(datapoints[x][0]/datapoints[x][1])
# 	# 	if(x in datapoints_g):
# 	# 		datapoints_g[x]=(datapoints_g[x][0] +datapoints[x][0]/datapoints[x][1] ,datapoints_g[x][1]+1)
# 	# 	else:
# 	# 		datapoints_g[x]=(datapoints[x][0]/datapoints[x][1],1)
# 	# print(X)
# 	# print(Y)

# 	# dict_index={}
# 	# for i in range(len(X)):
# 	# 	dict_index[X[i]]=Y[i]

# 	# X.sort()
# 	# Y=[]
# 	# for i in X:
# 	# 	Y.append(dict_index[i])



# 	# plt.plot(X, Y)

# 	final[(key, date)] = NoOfItems


# cumulative = {}
# for (restaurant, date) in final.keys():
# 	NoOfItems = final[(restaurant, date)]
# 	if date in cumulative:
# 		for i in range(0, 48):
# 			cumulative[date][i] += NoOfItems[i]
# 	else:
# 		cumulative[date] = NoOfItems


# # for x in datapoints_g.keys():
# # 	g_X.append(x)
# # 	g_Y.append(datapoints_g[x][0]/datapoints_g[x][1])


# # print(g_X)
# # print(g_Y)

# # dict_index={}
# # for i in range(len(g_X)):
# # 	dict_index[g_X[i]]=g_Y[i]

# # g_X.sort()
# # g_Y=[]
# # for i in g_X:
# # 	g_Y.append(dict_index[i])
	

# # plt.plot(g_X, g_Y)

# date = '2021-12-17'
# X = []
# y = []
# for i in range(0, 48):
# 	X.append(i/2)
# 	y.append(cumulative[date][i])

# plt.plot(X, y)
# plt.title("Total Orders on "+date)
# plt.xlabel('Time (in 24 hour format)')
# plt.ylabel('Total no. of orders at 30 minute time window')
# # plt.savefig('individual_restaurants.png')
# plt.savefig(date+'.png')




# # 	# print(final)
# # 	# break