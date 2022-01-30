import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

x1 = []
y1 = []
x2 = []
y2 = []

with open("barrier_width.csv", "r", encoding='utf8')  as csvfile:
    plots = csv.reader(csvfile, delimiter=",")
    for row in plots:
        x1.append(float(row[0]))
        y1.append(float(row[1]) * 10)

with open("barrier_height.csv", "r", encoding='utf8') as csvfile:
    plots = csv.reader(csvfile, delimiter=",")
    for row in plots:
        x2.append(float(row[0]))
        y2.append(float(row[1]) * 10)

x1 = np.array([x for x in x1], dtype="float")
y1 = np.array([y for y in y1], dtype="float")
x2 = np.array([x for x in x2], dtype="float")
y2 = np.array([y for y in y2], dtype="float")

# define x as 200 equally spaced values between the min and max of original x
x1new = np.linspace(x1.min(), x1.max(), 200)
x2new = np.linspace(x2.min(), x2.max(), 200)

# define spline with degree k=7
spl1 = make_interp_spline(x1, y1, k=3)
spl2 = make_interp_spline(x2, y2, k=3)
y1_smooth = spl1(x1new)
y2_smooth = spl2(x2new)

# create smooth line chart
fig, axs = plt.subplots(2)
fig.suptitle('Tunneling probability vs. barrier width and barrier height')
# start linear regression
coef = np.polyfit(x1,y1,1)
poly1d_fn = np.poly1d(coef) 
axs[0].plot(x1, poly1d_fn(x1), '--r') #'--k'=black dashed line, 'yo' = yellow circle marker
# end lineal regression
axs[0].plot(x1new, y1_smooth)
axs[0].set_xlabel('Barrier width')
axs[0].set_ylabel('Tunneling probability')

# start linear regression
coef = np.polyfit(x2,y2,1)
poly1d_fn = np.poly1d(coef) 
axs[1].plot(x2, poly1d_fn(x2), '--r') #'--k'=black dashed line, 'yo' = yellow circle marker
# end lineal regression
axs[1].plot(x2new, y2_smooth)
axs[1].set_xlabel('Barrier height')
axs[1].set_ylabel('Tunneling probability')

plt.show()
fig.savefig('tunneling_probability.png')
