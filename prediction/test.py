import matplotlib.pyplot as plt
import sys



city = sys.argv[1]
month = sys.argv[2]

X_t = [1,2,3,4]
Y_t = [4,5,8,10]
plt.bar(X_t, Y_t)
plt.xticks(X_t)
plt.xlabel("Store Id")
plt.ylabel("RMSE value of test set")
plt.title("Preparation time prediction")
plt.savefig(f'plots/{city}_{month}_1.png')
plt.clf()

X_axis = [10, 11, 13]
Y_axis = [1,2,3]

plt.bar(X_axis, Y_axis)
plt.xticks(X_axis, rotation='vertical')
plt.xlabel("Time slot")
plt.ylabel("RMSE value of test set")
plt.title("Preparation time prediction over different time slots")
plt.savefig(f'plots/{city}_{month}_2.png')
plt.clf()

X_t = [1,2,3,4]
Y_t = [3,4,6,11]
plt.bar(X_t, Y_t)
plt.xticks(X_t)
plt.xlabel("Store Id")
plt.ylabel("RMSE value of test set")
plt.title("Preparation time prediction")
plt.savefig(f'plots/{city}_{month}_3.png')
plt.clf()

X_t = [0,1,2,3]
Y_t_all = [4,5,8,10]
Y_t = [3,4,6,11]
y = []
for i in X_t:
    y.append((Y_t[i], Y_t_all[i]))

plt.plot(X_t, y)
plt.legend(['Restaurent Model Prediction', 'Single Wise Prediction'])

plt.xticks(X_t, rotation='vertical')
plt.xlabel("Store Id")
plt.ylabel("RMSE value of test set")
plt.title("Preparation time prediction")
plt.savefig(f'plots/{city}_{month}_4.png')
